# TMT-CUDA Architektur (Stand: Commits bis `730f6bb`)

Reines CUDA-C++ (kein PyTorch, kein Python zur Laufzeit). Byte-Modell:
Embedding (256 → dim), pro Layer Rekurrenz + MoE (+ optional MLA),
Byte-Decoder (256) + Stop-Head (1). Training per Fenster-TBPTT mit
AdamW, Checkpoints sind versioniert und exakt fortsetzbar.

## 1. Präzision und GEMM-Regeln

- **Compute bf16, Master fp32.** Jede lernbare Größe liegt als fp32-Master
  plus bf16-Arbeitskopie vor; Gradienten fp32 (dX-Pfade bf16); Adam-Momente
  fp32; EMA-Zielencoder fp32.
- **cuBLAS-Regel (elementweise verifiziert):** A- und B-Operand eines GEMM
  müssen **denselben Dtype** haben (gemischt bf16×fp32 → `NOT_SUPPORTED`).
  Erlaubt: bf16×bf16→bf16, bf16×bf16→fp32, fp32×fp32→fp32 — alle mit
  `COMPUTE_32F`. Deshalb liegen Probs/dS im MLA-Pfad in bf16, nur
  Akkumulatoren (O, m, l, dQ, LSE) in fp32.
- **Gewichtslayout wie `nn.Linear`:** W(N,K) row-major = [out,in].
  Row-major `Y(M,N) = X(M,K) @ W(N,K)^T` läuft als
  `(OP_T, OP_N, N, M, K, W, K, X, K, Y, N)` (llm.c-Muster). Alle
  vorwärts/rückwärts-Varianten (`linear_fwd/dX/dW`, alle MLA-Batched-GEMMs,
  Router-GEMMs) sind in dieser Form elementweise nachgerechnet; die
  Historie (`pv`-Revert, `dq`/`dk`-Fix, Router-`ldb`) dokumentiert die
  Stellen, an denen Alias-Denken falsch lag.

## 2. Parametrisierung

- `ParameterStore` (train.cu): alle Parameter in **einem Vektor**, referenziert
  **per Index, nie per Pointer** (Vektor-Reallokation invalidiert Referenzen).
  Jeder Eintrag: master/m/v/grad (fp32) + work (bf16) + Länge.
- Initialisierung (Seed-Default 1234, Host-RNG): Embed uniform ±0.05,
  Lineare Kaiming-uniform, Router normal(0, 0.02), Gamma 1, Beta 0,
  Decay aus Halbwertszeit-Plan (siehe 3), Gate 0.

## 3. Hierarchische gated Rekurrenz (cell.cu)

Pro Dimension und Zeitschritt, mit explizitem Eingangs-Carry:

```
a = sigmoid(decay + gate * x)      # lernbares Gate pro Dim (gated=1)
state = a * state + (1 - a) * x    # konvexe Mischung -> beschränkt
```

- `gated=0` schaltet das Gate ab (`a = sigmoid(decay)`), dann wie früher.
- **Zeitskalen-Hierarchie:** `decay` wird nicht mit 0/2.0 initialisiert,
  sondern aus geometrisch gestaffelten Halbwertszeiten
  `half_min=2 … half_max=512` über die Dimensionen:
  `a0 = exp(-ln2/half)`, Logit-Init `log(a0/(1-a0))`. Frühe Dims vergessen
  nach ~2 Bytes, späte nach ~512 — das Netz *startet* mit sortiertem
  Kurz-/Langzeitgedächtnis statt es erst lernen zu müssen.
- Forward persistent: ein Thread pro (Batch, Dim) loopt die ganze Sequenz
  in Registern (S1-Bench: ~1000 GB/s, 3 Launches pro Fenster).
- **Backward exakt im Fenster:** ein Thread läuft rückwärts,
  `dS_total[t] = dSnorm[t] + a·dS_total[t+1]`, dazu exakte
  Decay-/Gate-Gradienten inkl. Beitrag des **Eingangs-Carrys** (TBPTT-Grenze
  ist konstant, aber ihr Einfluss auf Position 0 wird differenziert).
  Per Finite-Differenzen gegen CPU-Referenz verifiziert
  (`architecture_test`, inkl. Carry ≠ 0, gated an/aus).

## 4. MoE (moe.cu)

- Top-k-Routing (Default E=1: dichter SiLU-Pfad ohne Router).
- Router-Logits fp32 direkt aus GEMM; Softmax + Top-k + Renorm pro Zeile.
- **Token-Dispatch:** Histogramm + Präfix-Offsets auf Host (E ≤ 16),
  Permutation per Atomics, pro Experte genau ein GEMM **nur über seine
  Tokens** (Gather/Scatter sind fused Elementar-Kernel, keine Atomics im
  Combine: ein Thread pro Zeile summiert seine k Slots).
- SiLU wird *im* Combine auf den Pre-Aktivierungen angewendet
  (kein Extraspeicher); Backward rekomputiert `silu`/`silu'` daraus.
- **Differenzierbare Regularisierer:** Switch-Aux
  `E·Σ(mean_p·frac)` und z-Loss `mean(logsumexp²)` stehen nicht nur als
  Loss-Skalare, sondern mit **analytischen Router-Gradienten**
  (`router_bwd_kernel`: Softmax-Rückweg plus Aux-Term
  `aux·E/N·p·(frac−erwartet)` und z-Term `zcoef·2/N·lse·p`).
- Verifiziert: Forward + alle Grade (dX, dRouter, dExperten) gegen
  Torch-Autograd; Aux/z-Gradienten per Finite-Differenzen.

## 5. MLA-128k (mla.cu, Standard: aus)

- DeepSeek-Prinzip ohne Absorptions-Trick (exakt, verifizierbar): pro Position
  nur **komprimierter Latent (Rank L) + entkoppelter RoPE-Schlüssel (Dim R)**
  im Ringpuffer. Beispiel d=1280, L=128, R=64: 128k·192·2 B ≈ 50 MB pro
  Stream und Layer.
- Pro Layer und Köpfe: Q aus `Wq`, Up-Projektionen `Wuk/Wuv` pro Chunk,
  RoPE (NeoX-Paare, Theta-Default 10000) auf Queries (mit Fensterposition)
  und Keys (beim Schreiben gebacken).
- **Chunked Online-Softmax** (Chunk-Default 256/1024/2048): Scores und
  P·V als gestapelte Batched-GEMMs über alle Köpfe×Streams (ein Launch pro
  Chunk und Operation), `(m, l)`-Akkumulatoren fp32, Kausal-Maske über
  globale Positionen, LSE pro Query für den Backward.
- **Backward per Chunk mit Recompute** (kein P-Cache über Chunks):
  S-Recompute → dP → Softmax-Backward (P wird in den toten S-Puffer
  geschrieben) → dQ-Akku, dKc/dVc, Up-Grade via `dlat_gemm` mit Beta-Akku,
  Rope-Backward (negierte Winkel), maskierter Scatter nur für Fenster-
  Positionen (Vergangenheit bekommt keinen Gradienten — TBPTT-Semantik).
- **Reset-on-full:** passt das Fenster nicht mehr in `mla_cache`, wird der
  Ring zurückgesetzt (linearer Slot↔Positions-Mapping bleibt trivial);
  Position des Resets steckt im Checkpoint.
- Verifiziert: Forward + **alle** Grade gegen unabhängige FP64-CPU-Referenz,
  inkl. partielle Chunks, Multi-Chunk-Grade, RoPE-Positionen, Präfix-Eviction.

## 6. Streaming-State und shared Forward-Pfad

- `StreamState` (model.cu) ist vom Modell **entrennt**: Carry-Vektor pro
  Layer, MLA-Ring-Caches pro Layer, Positionszähler. `build_state` +
  `reset_state` verwalten Allokation/Nullen.
- **Train und Eval nutzen denselben Forward-Pfad** (`forward_window`):
  pro Layer Input-Snapshot + Carry-In-Snapshot sichern, State-Loop,
  Norm, MoE, optional MLA, Residual, Carry-Out extrahieren.
  `backward_window` verlangt Forward unmittelbar davor (Workspace-Sharing).
- Fenster-TBPTT: Carry wird nur an Fenstergrenzen detached; `maxcarry`
  (Default 0 = aus) begrenzt Drift zusätzlich; neue Epoche/Datei resettet.

## 7. Losses und Optimizer

- `loss = var·hinge + latent·MSE + ce·CE + stop·BCE + (aux+z)/layers`.
  Defaults: **CE-only** (`latent=var=stop=0`), CE-Pflicht (`ce>0` validiert).
- Stop als rohe Logits mit `BCEWithLogits + pos_weight` (Default 20, EOS selten);
  Latent-MSE gegen **EMA-Zielencoder** (kein Selbst-Jagen); Varianz-Hinge
  gegen Kollaps; alle Gewichte einzeln schaltbar.
- AdamW (β 0.9/0.999, eps 1e-8, wd 0.01), globaler Grad-Clip über
  `Snrm2`-Summe, Warmup+Cosine-Schedule, **NaN-Schutz** (nicht-finiter
  Gradient → Update verweigert statt Gewichte zu vergiften), EMA-Update
  inkl. bf16-Refresh der Arbeitskopie. Stop-Head ohne Gewicht wird beim
  Update übersprungen.

## 8. Checkpoints V3 (checkpoint.h, inkompatibel zu V2)

Format `TMTCPKT3`: Magic + **Config als Text** (Schema-Roundtrip-geprüft,
Mismatch → Abbruch statt still falschem Fortsetzen) + Fortschritt
(`step/cursor/epoch/carried`, Daten-Hash/Größe) + pro Parameter
(master/m/v) + Stream-Position + Carries + MLA-Cache-Köpfe und
-initialisierte Inhalte + **FNV-Checksumme**, geschrieben per tmp+fsync+rename.
Laden verifiziert erst die Checksumme (Trunkierung/Korruption → Abbruch),
stellt Gewichte **und** Stream-Historie wieder her. CLI-Tests beweisen:
Split-vs-Whole-Resume ist bis auf FP32-Toleranz identisch, Eval verändert
die Datei nicht (`cmp`), falsche Config/Daten/Trunkierung wird abgewiesen.

## 9. CLI und Daten

- `train DATA CKPT [mode=train|eval] [steps=N] [saveevery=N] [key=value …]`;
  existiert ein Checkpoint, liefert er die Config (CLI-Overrides müssen
  passen). Daten per `mmap`, in B Shards gestaffelt, Fenster der Länge T.
- `mode=eval` schreibt JSON pro Fenster (`bytes/ce/bpb`) ohne zu lernen;
  SIGINT-Flag für sauberen Abbruch.

## 10. Verifikation (lokal bestanden, RTX 5070)

- `architecture_test`: finite Differenzen für gated/ungated Cell (mit Carry),
  MoE-Aux/z-Gradienten, Lern-Smoke (CE 5.42→0.75 dense, →0.82 MoE),
  Stream-Trennung, Resume, Korruptions-Abweisung, MLA-Teilchunks/
  Multi-Chunk-Grade/RoPE/Eviction, MLA-vs-FP64-Referenz.
- `tests/cli.sh`: Resume-Determinismus (dense + MLA), Eval-Tails,
  unveränderte Checkpoints, Ablehnung invalider Inputs.
- `compute-sanitizer`: 0 Speicherfehler (Stand der Commit-Botschaft).

## 11. Dateiübersicht (tmt-cuda/src)

| Datei | Inhalt |
|---|---|
| `config.h` | Einziges Config-Schema (Macro-Tabelle), Parsen, Validierung |
| `common.h`/`util.h` | bf16-Helfer, `CUDA_CHECK` (ohne Shadowing-Falle), cuBLAS-Handle |
| `linalg.cu` | bf16-GEMM-Wrapper fwd/dX/dW (dW mit Beta-Akku) |
| `cell.cu` | Gated hierarchische Cell fwd/bwd + S1-Demo-Pfad |
| `norm.cu` | LayerNorm mit Gamma/Beta fwd/bwd |
| `moe.cu` | Dispatch, Top-k, Combine±SiLU, analytische Aux/z-Grade, E=1-Pfad |
| `mla.cu` | Latent-Ring, Chunk-Attention fwd/bwd, RoPE, Positions-Masken |
| `emb.cu`/`loss.cu` | Gather/Atomic-Scatter, CE/Stop/Latent/Var-Kernel |
| `adam.cu` | Fused AdamW pro Parameter |
| `train.cu` | `ParameterStore`, `DeviceMemory`, alte Harness-Reste |
| `model.cu` | Modell/State-Aufbau, `forward_window`/`backward_window`, Optimizer |
| `checkpoint.h` | V3-Format, Checksummen, Resume-Validierung |
| `train_main.cu` | CLI, mmap-Daten, Main-Loop, SIGINT |
| `architecture_test.cu` | Native Regressionstests (s.o.) |
| `bench.cu` | S1-Bandbreiten-Demo (~1000 GB/s) |

## 12. Offen vor dem großen Run

1. **Speicherrechnung Prod-Config** (420M-Params/128k-Cache/16 GB VRAM):
   Cache allein ≈ 12 GB bei 32 Layern × 16 Streams — passt nicht mit
   Adam (~5 GB) + Aktivierungen zusammen. Optionen: weniger Streams,
   Cache 64k, MLA nur in jedem n-ten Layer (`mla_every`).
2. **Sampler:** Eval misst nur CE/BPB; freie Generierung (Prompt → Bytes)
   fehlt für Qualitätskontrolle im Langlauf.
3. **FP8** erst nach Messung (Matmul-limitiert?); **Compile/CUDA-Graphs**
   als Durchsatz-Hebel nachgelagert.
4. **Daten:** enwiki-Dump (Download läuft auf Fisch) entpacken/splitten;
   Harness liest derzeit ≤ 8 GB per `fread` (Streaming für 90 GB XML fehlt).
