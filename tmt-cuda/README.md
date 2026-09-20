# TMT-CUDA

Aktiver CUDA-Entwicklungspfad des Byte-Modells. PyTorch (`main_torch_moe.py`,
`benchmark_torch_moe.py`) ist veraltet; das MLX-Modell dokumentiert den ursprünglichen
Prototyp. Neue CUDA-Checkpoints sind mit beiden nicht kompatibel.

## Bauen und prüfen

Benötigt werden Linux, ein NVIDIA-Treiber, CUDA mit BF16/cuBLAS und ein C++17-Hostcompiler.
Die Makefile-Voreinstellung ist Consumer-Blackwell (`sm_120a`). Andere GPUs benötigen
passende `ARCH`-Flags und BF16-Unterstützung, beispielsweise:

```sh
cd tmt-cuda
make
make check
# Beispiel für eine andere Zielarchitektur:
make ARCH='-gencode arch=compute_89,code=sm_89'
# Nach Änderung von ARCH zuerst `make clean` ausführen.
```

`make check` benötigt eine GPU, aber weder Python noch PyTorch. Es prüft:

- normalisierte Rekurrenz mit/ohne Eingabe-Gate, nichtleerem Carry und numerischen Gradienten;
- Gradienten von MoE-Load-Balancing und Router-z-Loss;
- Abhängigkeit höherer Schichten von niedrigeren Experten und Lernfortschritt;
- getrennte Streaming-Zustände, Checkpoint-Fortsetzung und Korruptionserkennung;
- MLA mit mehreren/teilweisen Chunks, Cache-Verdrängung und nichtnull RoPE-Positionen;
- MLA-Forward und alle Gradienten gegen eine unabhängige FP64-CPU-Referenz;
- CLI-Training, Evaluation einschließlich Restfenster und ungültige Eingaben.

Zusätzliche Speicherprüfung:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 ./architecture_test
./bench 2 32 128
```

`bench` misst nur die rekurrente Zelle. Es ist kein Durchsatzbenchmark des gesamten
Trainings. Seine Bandbreitenangabe ist aus dem geschätzten Datenverkehr berechnet.

## Modell und Architekturentscheidungen

```text
Bytes → Embedding → [hierarchische Rekurrenz → LayerNorm → Dense/MoE + Residual] × L
                       optional nach ausgewählten Blöcken: LayerNorm → MLA + Residual
      → finale Repräsentation → Byte-Logits (256)
```

Jede Schicht erhält den Residualstrom **der vorherigen Schicht**. Der vorherige
Prototyp speiste alle rekurrenten Schichten direkt aus demselben Byte-Embedding.
Die neue Struktur ermöglicht hierarchische Merkmalsverarbeitung und führt den
Gradienten entsprechend durch alle vorherigen Schichten zurück.

Pro Kanal, Schicht und Byte gilt:

```text
a_t = sigmoid(decay + gate * x_t)
s_t = a_t * s_(t-1) + (1 - a_t) * x_t
x_out = x_t + FFN(LayerNorm(s_t))
```

Das normalisierte Update begrenzt die Akkumulation bei langsamen Decays. Das Gate
startet bei null und wird gelernt; `gated=0` erlaubt die statische Ablation.
Die initialen Halbwertszeiten sind logarithmisch von `half_min=2` bis `half_max=512`
Bytes verteilt. Bei einem eingabeabhängigen Gate ist die tatsächliche Zeitskala
anschließend kontextabhängig. Ein fortlaufender Zustand garantiert kein unbegrenztes
Erinnerungsvermögen.

Standard ist ein kleines dichtes Modell: `dim=256 layers=4 experts=1 topk=1`.
Der dichte Pfad führt keine Routing-Operationen aus. Für `experts>1` werden nur
zugewiesene Token-Experten-Paare berechnet. Der Load-Balancing-Loss verwendet die
Zuordnungsanteile `counts / (N * topk)`; sein Gradient und der z-Loss fließen in
den Router ein. MoE-Dispatch verwendet noch dynamische Allokationen und
Host-Synchronisation. Eine pauschale Beschleunigung durch MoE ist nicht belegt.

BF16 wird für Arbeitsgewichte/Aktivierungen und GEMMs verwendet. Rekurrente
Zustände, Mastergewichte, Adam-Momente und wesentliche Reduktionen sind FP32.
Das Training verwendet TBPTT: Der eingehende Carry ist an der Fenstergrenze
abgetrennt, trägt aber korrekt zum Decay-/Gate-Gradienten des ersten Bytes bei.

Next-Byte-Cross-Entropy ist das Standardziel. `latent=0 var=0 stop=0` deaktiviert die
zusätzlichen Zielanteile. Diese lassen sich für kontrollierte Ablationen aktivieren.
Der EMA-Zielencoder wird ausschließlich per EMA aktualisiert, nicht durch AdamW.
Der optionale Stop-Loss benutzt das nächste Newline-Byte als Ziel; dies ist keine
allgemeine Dialog-Endemarkierung.

## Kleine gemeinsame Schnittstelle

| Datei | Verantwortung |
|---|---|
| `src/config.h` | Ein Konfigurationsschema für CLI, Validierung und Checkpoints |
| `src/train.cu` | Gemeinsame CUDA-Operatoren und modellbezogener Parameterspeicher |
| `src/model.cu` | Modellaufbau, expliziter Zustand, Forward, Backward und Optimizer-Schritt |
| `src/checkpoint.h` | Versionierte Speicherung, Prüfung und Wiederherstellung |
| `src/train_main.cu` | Datenzugriff, Trainings-/Evaluationsablauf und Messausgabe |
| `src/architecture_test.cu` | Native numerische und Integrationstests |

Die interne CUDA-Schnittstelle ist bewusst klein:

```cpp
build_model(model);
build_state(state, model);
forward_window(model, state, loss, ce); // aktualisiert ausschließlich den übergebenen Stream
// model.X: finale Repräsentationen [B,T,D]; model.logits: [B,T,256]
backward_window(model, state);         // nur beim Training, direkt nach Forward
optimizer_step(model, step);
release_window(model);                // Forward-Scratch nach Nutzung freigeben
reset_state(state, model.c);           // Gewichte bleiben unverändert
```

Mehrere Modelle haben getrennte Parameterspeicher. Mehrere `StreamState`-Objekte
können dasselbe Modell nacheinander verwenden. Der Forward-Arbeitsspeicher gehört
zum Modell; gleichzeitige Aufrufe auf demselben Modell sind nicht unterstützt.
Forward muss vor einem Backward unmittelbar auf demselben Modell gelaufen sein.
Repräsentationen werden beim nächsten Forward überschrieben und müssen bei Bedarf
kopiert werden. Das erlaubt später korrekte Klassifikations-Probes auf dem tatsächlichen
Modellausgang. Ein CUDA-CoLA-Adapter ist noch nicht implementiert.

## Training und Evaluation

Die Datendatei enthält rohe Bytes, ohne Tokenizer. `mmap` erlaubt dateiweises
Paging durch das Betriebssystem; es gibt keine bisherige 8-GB-Dateigrenze und
keine Begrenzung auf das erste MiB. Die Datei muss während eines Laufs unverändert
bleiben. Für die Fortsetzung wird ihre Identität anhand von Größe und Inhaltshash geprüft.

```sh
# Neuer Lauf; begrenztes Budget empfohlen, steps=0 läuft bis zum Abbruch.
./train train.bin model.ckpt steps=1000 saveevery=100

# Konfiguration, Adam, Datenposition und Streaming-Zustand werden geladen.
# steps zählt zusätzliche Schritte, nicht die globale Zielschrittzahl.
./train train.bin model.ckpt steps=1000

# Gewichte unverändert; frischer Stream, keine Checkpoint-Schreibzugriffe.
./train validation.bin model.ckpt mode=eval

# Eigenständiges MoE-Experiment, gleicher Daten-/Budgetvergleich erforderlich.
./train train.bin moe.ckpt experts=4 topk=2 steps=1000

# Ablationen jeweils mit eigenem Checkpoint-Pfad und festem Seed.
./train train.bin static.ckpt gated=0 steps=1000
./train train.bin latent.ckpt latent=1 steps=1000
```

Bestehende Checkpoints liefern ihre Konfiguration automatisch. Abweichende
Modell- oder Trainingsparameter werden beim Fortsetzen abgewiesen. Für eine
andere Konfiguration wird ein neuer Checkpoint-Pfad verwendet. `mode`, `steps`
und `saveevery` sind Laufsteuerung und gehören nicht zur gespeicherten Konfiguration.
`saveevery=0` speichert nur am Ende. SIGINT/SIGTERM beendet nach dem aktuellen
Fenster und speichert beim Training; bei einem Prozessabsturz bleibt der letzte
vollständige Checkpoint erhalten.

Wichtige Optionen:

| Optionen | Standard / Bedeutung |
|---|---|
| `dim`, `layers`, `experts`, `topk` | `256`, `4`, `1`, `1` |
| `batch`, `seqlen` | `8`, `128`; TBPTT-Fenster in Bytes |
| `gated`, `half_min`, `half_max` | `1`, `2`, `512` |
| `lr`, `warmup`, `decaysteps`, `minlr` | `0.0005`, `200`, `8000`, `0.1` |
| `ce`, `latent`, `var`, `stop` | `1`, `0`, `0`, `0` |
| `aux`, `zloss` | `0.01`, `0.001`; nur bei MoE |
| `gradclip`, `ematau`, `seed` | `1`, `0.99`, `1234` |
| `maxcarry` | `0`: kein periodischer Reset; positive Werte: Reset zwischen Fenstern |

Die Datei wird in `batch` zusammenhängende Streams aufgeteilt. Jeder Stream
startet mit leerem Zustand. Training verarbeitet vollständige Fenster; am Ende
einer Epoche werden Restbytes verworfen und alle Zustände zurückgesetzt. Die
Evaluation verarbeitet auch Restfenster; Padding geht nicht in CE/BPB ein.
Übergänge zwischen den Streams werden nicht bewertet. Vergleiche müssen daher
dieselbe Aufteilung und denselben Reset-Modus verwenden.

Die letzte Ausgabezeile ist ein JSON-Objekt mit `mode`, `steps`, `bytes`, `ce`,
`bpb`, `seconds` und `bytes_per_second`. BPB bedeutet Bits pro Byte (`CE / ln(2)`),
nicht Perplexität pro Subword-Token. Die Trainingsmessung enthält Forward, Backward,
Updates und gegebenenfalls periodische Checkpoints; Initialisierung und der letzte
Checkpoint liegen außerhalb der Zeitmessung. Evaluation enthält Forward und
Scoring. Ein Lauf mit `steps>0 mode=eval` bewertet nur einen Präfix.

## Checkpoints und Reproduzierbarkeit

Format V3 speichert die vollständige Konfiguration, FP32-Mastergewichte,
Adam-Momente, EMA-Encoder, globalen Optimizer-Schritt, Datenposition/Epoche,
Datensatzfingerabdruck, rekurrente Zustände und den gültigen MLA-Cache einschließlich
absoluter Positionen. BF16-Arbeitsgewichte werden daraus rekonstruiert.

Dateien werden über eine temporäre Datei mit Flush und atomarem Rename ersetzt.
Ein Inhaltschecksum wird vor der Wiederherstellung geprüft. Alte V2-, MLX- und
PyTorch-Checkpoints werden nicht stillschweigend als neue Modelle geladen.
V3 ist ein natives Linux-64-Bit-Binärformat, kein plattformunabhängiges Austauschformat.
Der Hash dient der Erkennung versehentlicher Änderungen, nicht der Authentifizierung.

Seed und Datenfortschritt sind reproduzierbar. CUDA-Atomics und cuBLAS können
kleine Rundungsunterschiede verursachen; bitidentische Ergebnisse über beliebige
GPUs, Treiber oder Dispatch-Reihenfolgen sind nicht garantiert. Fortsetzung wird
numerisch gegen einen ununterbrochenen Update geprüft.

## Optionaler MLA-Cache

```sh
./train train.bin attention.ckpt mla=1 mla_every=2 mla_cache=4096 mla_cc=256 steps=1000
```

MLA ist standardmäßig deaktiviert. Der Cache speichert je aktivem Attention-Block
und Stream komprimierte KV-Latents plus RoPE-Keys. Die reine Cachegröße beträgt:

```text
batch × Anzahl MLA-Blöcke × mla_cache × (mla_L + mla_R) × 2 Bytes
```

Gewichte, Optimizer, Aktivierungen und Attention-Arbeitsspeicher kommen hinzu.
Die Attention liest den gesamten gültigen Cache in Chunks. Ihre Rechenkosten
wachsen deshalb mit der Kontextlänge trotz komprimierter Speicherung.

Bei Überlauf wird nur der älteste nötige Präfix entfernt. Die Verdrängung geschieht
vor einem ganzen Fenster: Das erste Byte dieses Fensters hat bis zu `seqlen-1`
ältere Positionen weniger zur Verfügung als bei einem strikt byteweisen Sliding
Window. Absolute RoPE-Positionen bleiben erhalten. Vergangene Cache-Einträge
sind an der TBPTT-Grenze abgetrennt; die Up-Projektionen erhalten weiterhin Gradienten.

`mla_cache=131072` ist konfigurierbar, aber weder Abrufqualität noch Durchsatz bei
128k Bytes sind durch die kleinen Regressionstests belegt. Der Cache ist kein
verlustfreies Archiv und es gibt keine Garantie eines „exakten 128k-Abrufs“.

## Plan: RSI-Learning über mehrere Aufgabenfamilien

**Status: geplant, kein automatischer RSI-Controller implementiert.** Das aktuelle
CUDA-Training und die unverändernde Evaluation bilden die ausführbare Grundlage.

[Dream-RSI](https://arxiv.org/html/2609.14858v1) verbessert im Paper die ausführbare
Explorationsstrategie um einen festen Agenten. Historische Versuchsbäume dienen
als Replay-Welten; neue Strategien entscheiden über Fortsetzung, Verzweigung und
Abbruch. Das Verfahren ersetzt kein Gewichtslernen. Replay deckt nur tatsächlich
beobachtete Fortsetzungen ab. Verbesserungen auf alten Bäumen garantieren keinen
Transfer auf neue Aufgaben. Die folgende Übertragung auf TMT ist ein Projektplan.

### 1. Verlässliche Aufgaben und Baselines

Ein fester Evaluator erhält ausschließlich ein Modellartefakt, ein versioniertes
Aufgabenmanifest und ein Budget. Alle Modelländerungen passieren außerhalb des
Evaluators. Zuerst werden folgende CUDA-Adapter umgesetzt:

| Familie | Aufgaben / Protokoll | Zielgröße |
|---|---|---|
| Sprache | Zurückgehaltene Texte aus mehreren Quellen | BPB je Quelle |
| Grammatik | CoLA-Probe auf finaler Repräsentation; BLiMP-Satzwahrscheinlichkeiten | MCC / Paar-Genauigkeit |
| Gedächtnis | Copy, verzögerter Abruf, Schlüssel-Wert-Zuordnung | Genauigkeit nach Distanz |
| Algorithmen | Addition, Klammerprüfung, kleine Zustandsautomaten | Exakte Lösung, längere Eingaben |
| Fortlaufendes Lernen | Domänenwechsel und Rückkehr zu alten Aufgaben | Anpassung und Vergessen |
| Ressourcen | Feste Shapes und Warmup-Regeln | Laufzeit, Bytes/s, GPU-Spitzenspeicher |

[BLiMP](https://github.com/alexwarstadt/blimp) ist eine Sammlung grammatischer
Minimalpaare. Bei Byte-Modellen muss die vollständige Satzwahrscheinlichkeit
mit identischer Zustandsinitialisierung verglichen werden.

Trainingsdaten, Entwicklungsdaten und gesperrte Testdaten werden getrennt.
Generierte Aufgaben bekommen getrennte Seeds und zusätzlich zurückgehaltene
Längen/Strukturen. Ganze Aufgabenfamilien werden für Transfertests ausgeschlossen.
Testdaten dürfen weder in Prompts noch in Strategieauswahl oder Gewichtsupdates gelangen.

Abnahme: reproduzierbare Manifeste und Einzelmesswerte, mindestens drei Seeds,
Vergleich der dichten Baseline mit statischer/gesteuerter Rekurrenz bei gleichem
Budget. Die Tests im Repository ersetzen diese Qualitätsbenchmarks nicht.

### 2. Einheitliches Experimentprotokoll

Ein kleiner Runner (Rust ist für Prozessverwaltung und Ablage vorgesehen) startet
CUDA-Prozesse mit begrenzter GPU-Zeit und sammelt deren JSON-Ergebnisse. Es gibt
zunächst eine lokale Job-Warteschlange statt einer verteilten Servicearchitektur.
Ein Aufgabenadapter liefert über dieselbe Schnittstelle Eingaben und überprüfbare
Ergebnisse; pro Benchmark entsteht kein eigenes Trainingssystem.

Jeder Versuch protokolliert:

- Versuch-ID, Eltern-ID, Code-/Build-Hash und vollständige Konfiguration;
- Daten-/Aufgabenmanifest, Seeds und Hardware-/Treiber-Version;
- Ausgangs- und Endcheckpoint einschließlich Optimizer-/Streaming-Zustand;
- alle Einzelmetriken, Fehler und Abbrüche;
- GPU-Zeit, Peak-Speicher sowie Agenten-Tokens/-Kosten.

Das spätere Controller-Interface lautet konzeptionell:

```text
select(observed_history, remaining_budget) -> parent_ids
execute(parent_id, experiment_spec, budget) -> observation + artifact
```

Die leere Auswahl beendet eine Suche. Zunächst werden nur kompatible
Trainingsfortsetzungen aus einem Checkpoint unterstützt. Eine geänderte Architektur
startet als neuer Wurzelversuch; das heutige strenge Checkpoint-Laden wird dafür
nicht aufgeweicht. Eine Curriculum-Erweiterung muss Datensatzwechsel explizit
protokollieren und bekommt einen separaten Import-/Fortsetzungsvertrag.

Abnahme: wiederholbare Wiederaufnahme nach Abbruch, unveränderlicher Evaluator,
gleiche Budgets für alle Kandidaten und vollständig nachvollziehbare Herkunft.

### 3. Feste Suche vor lernender Suche

Erst zufällige Suche, eine feste Verzweigungsstrategie und Successive Halving als
Vergleich implementieren. Klein beginnen: Gates, Zeitskalen, Lernrate,
Fensterlänge und Loss-Ablationen. MoE/MLA erst nach stabilen Baselines hinzufügen.
Ein vorhandener Coding-Agent kann später Änderungen vorschlagen; dass das kleine
TMT selbst solchen Code schreiben kann, ist bisher nicht nachgewiesen.

Qualität wird je Aufgabenfamilie auf vorab festgelegte Baselines normiert. Familien
werden gleich gewichtet, nicht nach Anzahl ihrer Untertests. Einzelwerte bleiben
sichtbar. Eine mögliche Auswahlregel lautet:

```text
score = mean(normalized_family_scores) - lambda * normalized_total_cost
```

Kosten umfassen Online-Versuche UND Strategieentwicklung. Ressourcenlimits,
Korrektheit und maximal erlaubte Regressionen sind harte Zulassungsbedingungen.
Die Gewichte/Normierung werden vor dem Experiment festgelegt, nicht nach Sichtung
der Testergebnisse angepasst.

### 4. Historisches Replay und reale Validierung

Aus protokollierten Versuchen entstehen versionierte Replay-Bäume. Jede alternative
Strategie startet mit leerem Beobachtungszustand. Der Controller sieht ausschließlich
bereits freigelegte Ergebnisse. Weder zukünftige Scores noch ausgeblendete Zweige
werden als Features zugänglich gemacht.

Eine aufgezeichnete Fortsetzung wird nur dann wiederverwendet, wenn Elternartefakt,
Aktion und Ausführungskontext übereinstimmen. Fehlende Fortsetzungen heißen
„unbekannt“; ihnen wird kein erfundener Erfolg oder Misserfolg zugeordnet. Neue
Aktionen müssen online ausgeführt werden. Die Suche enthält weiterhin neue
Wurzelversuche, um die historische Abdeckung zu erweitern.

Strategien werden auf getrennten historischen Bäumen entwickelt und validiert.
Der beste Replay-Kandidat tritt anschließend online gegen die bisherige Strategie
an. Er wird nur bei einem belastbaren Vorteil unter gleichem Gesamtbudget übernommen.
Messgrößen: Qualität bei festem Budget, Kosten bis zu einer Zielqualität, Regressionen
und Transfer auf zurückgehaltene Aufgabenfamilien.

### 5. Tatsächliches Lernen der Modellgewichte

Die ausgewählten Trainingsläufe verbessern TMT per Gradientenlernen. Überprüfbare
synthetische Aufgaben liefern Eingabe-Ziel-Paare; korrekte Agentenlösungen können
nach Prüfung und Deduplikation als zusätzliche Trainingsdaten dienen. Frühere
Domänen bleiben in einer festgelegten Mischung erhalten, um Vergessen zu messen
und zu begrenzen. Ein Benchmark-Gesamtscore allein ist kein ausreichendes
supervisiertes Trainingssignal.

Reinforcement Learning wäre ein gesondertes Experiment mit festem Verifier,
Policy-/Referenzcheckpoint und unabhängiger Evaluation. Es gehört nicht zur ersten
Dream-RSI-Integration. Zuerst muss die Kombination aus überwachten Aufgaben,
reproduzierbaren Experimenten und gelernter Suchsteuerung die festen Baselines schlagen.

## Stand der Verifikation

Die nativen Architekturtests und die CLI-Tests wurden auf einer NVIDIA GeForce
RTX 5070 Laptop GPU ausgeführt. CUDA Compute Sanitizer (`memcheck`) meldete für
die Architekturtests keine Fehler. Dies bestätigt die geprüften kleinen Shapes
und Rechenwege; Modellqualität auf den geplanten Benchmarks und Skalierung auf
128k Bytes sind weiterhin offen.
