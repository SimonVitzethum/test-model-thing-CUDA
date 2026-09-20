# TMT-CUDA — Training ohne PyTorch, direkt auf CUDA

Ziel: recurrentes Byte-Modell (MoE) in BF16 trainieren, mit persistenten
Kernen (ein Launch pro Fenster statt tausende), Tensor Cores für alle
Matmulen (cuBLAS), 128k Abruf-Kontext via MLA-Latent-Cache nach DeepSeek.

## Architektur (Kernentscheidung)

Ursprungsziel „kein Kontextlimit" bleibt — zweigeteilt:

- **Recurrence = unbegrenzt.** Der Decay-State läuft ewig weiter (Streaming,
  O(1) Speicher). Das ist das Arbeitsgedächtnis ohne Limit.
- **MLA = exakter Abruf über die letzten 128k Tokens.** Statt vollem KV-Cache
  (128k × L × d × 2B — viel zu groß) wird pro Position nur ein komprimierter
  Latent-Vektor im Ringpuffer gehalten (DeepSeek-V3-Prinzip: kv_lora_rank).
  Rechnung für d=1280, rank=128: 128k × 128 × 2B = 32 MB. Passt locker.

```
Byte -> Embed -> [Recurrent Cell + MoE]xL -> MLA-128k (latent Ring) -> Head
                      ^ unbegrenzt          ^ letzte 128k exakt
```

## Präzision

- Compute: BF16 (FP16 per Flag möglich, identischer Codepfad).
- Master-Gewichte + Adam (m,v): FP32. Kein FP8 in Stufe 1–3 (erst wenn
  Matmul-limitiert gemessen, nicht vorher).

## Stufen

- **S1 (dieser Commit):** fused persistente bf16 Recurrent-Cell
  (state+stats+norm/out, 3 Launches pro Fenster), Bench-Harness mit
  Roofline (GB/s vs. ~960 GB/s der 5080) + CPU-Referenzcheck.
- **S2:** MoE-Experten + Router als cuBLAS-GEMMs (Tensor Cores), Adam als
  fused Kernel, Fenster-Training end-to-end, Checkpoint-Format.
- **S3:** MLA-128k (compress/write/read, Ringpuffer), 128k-Rollout-Nachweis.
- **S4:** FP8 nur nach Messung (torchao-Äquivalent in Handarbeit), sonst nie.

## Bauen (Fisch-PC, CUDA 12.8, sm_100a Blackwell)

```
cd tmt-cuda && make            # ./bench
./bench [B T D]                # default 16 128 1280
```
