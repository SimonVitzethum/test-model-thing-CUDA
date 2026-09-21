#!/bin/sh
# End-to-end tests of the CUDA executable; no Python dependency.
set -eu
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
awk 'BEGIN { for (i=0;i<1000;i++) printf "ABC" }' > "$work/data"
./train "$work/data" "$work/whole" dim=16 layers=2 batch=2 seqlen=7 steps=4 saveevery=0 > "$work/whole.log"
./train "$work/data" "$work/split" dim=16 layers=2 batch=2 seqlen=7 steps=2 saveevery=0 > "$work/split.log"
./train "$work/data" "$work/split" steps=2 saveevery=0 >> "$work/split.log"
./architecture_test --compare "$work/whole" "$work/split"
cp "$work/split" "$work/saved"
printf 'ABCDEFGHIJKLMNOPQRSTUVWXYZ12345' > "$work/heldout"
./train "$work/heldout" "$work/split" mode=eval > "$work/eval.log"
# 31 bytes partitioned into two streams: 29 next-byte pairs, including tail.
grep '"bytes":29,' "$work/eval.log"
cmp "$work/split" "$work/saved"
if ./train "$work/data" "$work/split" dim=32 steps=1 > "$work/error.log" 2>&1; then
    echo 'FAIL: mismatching configuration accepted'; exit 1
fi
if ./train "$work/heldout" "$work/split" steps=1 > "$work/error.log" 2>&1; then
    echo 'FAIL: mismatching resume dataset accepted'; exit 1
fi
head -c 123 "$work/split" > "$work/truncated"
if ./train "$work/data" "$work/truncated" steps=1 > "$work/error.log" 2>&1; then
    echo 'FAIL: truncated checkpoint accepted'; exit 1
fi
if ./train "$work/data" "$work/invalid" experts=17 steps=1 > "$work/error.log" 2>&1; then
    echo 'FAIL: invalid expert count accepted'; exit 1
fi
# Cache overflow, partial attention chunks and exact state resume with MLA.
./train "$work/data" "$work/mla" dim=16 layers=2 batch=2 seqlen=7 mla=1 mla_heads=2 mla_dh=4 mla_L=4 mla_R=6 mla_cache=17 mla_cc=3 steps=3 saveevery=0 > "$work/mla.log"
./train "$work/data" "$work/mla" steps=1 saveevery=0 >> "$work/mla.log"
./train "$work/data" "$work/mla-whole" dim=16 layers=2 batch=2 seqlen=7 mla=1 mla_heads=2 mla_dh=4 mla_L=4 mla_R=6 mla_cache=17 mla_cc=3 steps=4 saveevery=0 > "$work/mla-whole.log"
./architecture_test --compare "$work/mla" "$work/mla-whole"
cp "$work/mla" "$work/mla-saved"
./train "$work/heldout" "$work/mla" mode=eval > "$work/mla-eval.log"
cmp "$work/mla" "$work/mla-saved"
# Hybrid traces: exact resume of trace state, and the sampler skips it.
./train "$work/data" "$work/tr" dim=16 layers=2 batch=2 seqlen=7 traces=1 steps=2 saveevery=0 > "$work/tr.log"
./train "$work/data" "$work/tr" steps=2 saveevery=0 >> "$work/tr.log"
./train "$work/data" "$work/tr-whole" dim=16 layers=2 batch=2 seqlen=7 traces=1 steps=4 saveevery=0 > "$work/tr-whole.log"
./architecture_test --compare "$work/tr" "$work/tr-whole"
./sample "$work/tr" "AB" maxlen=8 > "$work/tr-sample.log"
echo 'PASS CLI: dense resume within FP32 tolerance, evaluation tails, unchanged checkpoints, invalid input rejection, MLA resume and trace resume'
