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
# Evaluation may override the window and the carry limit (context ablation).
./train "$work/heldout" "$work/split" mode=eval seqlen=3 maxcarry=6 > "$work/eval-override.log"
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
# Document resets, trace decay and diagnostics; gradient comparison tool.
./train "$work/data" "$work/doc" dim=16 layers=2 batch=2 seqlen=7 traces=1 trace_decay=0.99 docsep=65 steps=100 saveevery=0 > "$work/doc.log"
grep -q '^traces: decay' "$work/doc.log"
grep -q '^state:' "$work/doc.log"
./train "$work/heldout" "$work/doc" mode=eval > "$work/doc-eval.log"
./gradcheck "$work/data" "$work/doc" len=28 window=7 seqs=2 > "$work/gradcheck.log"
grep -q '^decay' "$work/gradcheck.log"
# Optional loss terms (latent, variance, stop) still train when enabled.
./train "$work/data" "$work/losses" dim=16 layers=2 batch=2 seqlen=7 latent=1 var=1 stop=1 steps=3 saveevery=0 > "$work/losses.log"
grep -q '"mode":"train"' "$work/losses.log"
# Dialog data, answer-only loss, init= from another checkpoint, interactive chat.
awk 'BEGIN { for (i=0;i<200;i++) printf "\036\002hi\004\003ok\004" }' > "$work/dialog.bin"
./train "$work/dialog.bin" "$work/dlg" init="$work/whole" dialog=1 batch=1 seqlen=9 steps=40 saveevery=0 > "$work/dlg.log"
./train "$work/dialog.bin" "$work/dlg" mode=eval > "$work/dlg-eval.log"
grep -q '"bytes":600,' "$work/dlg-eval.log"   # 200 x ("o", "k", end of turn)
if ./train "$work/dialog.bin" "$work/dlg" init="$work/whole" steps=1 > "$work/error.log" 2>&1; then
    echo 'FAIL: init= accepted for an existing checkpoint'; exit 1
fi
printf 'hello\n/reset\nbye\n' | ./chat "$work/dlg" maxlen=6 > "$work/chat.log" 2> /dev/null
test "$(grep -c '^tmt> ' "$work/chat.log")" -eq 2
grep -q 'state reset' "$work/chat.log"
printf 'ABC\n' | ./chat "$work/whole" maxlen=4 > "$work/raw.log" 2> /dev/null
grep -q '^tmt> ABC' "$work/raw.log"
printf 'hi there\thello!\nno tab line\n' > "$work/pairs.tsv"
./dialogprep tsv "$work/pairs.tsv" "$work/pairs" test_frac=0 2> /dev/null
test "$(cat "$work/pairs_train.bin")" = "$(printf '\036\002hi there\004\003hello!\004')"
m() { printf '{"text":"%s","role":"%s","lang":"%s","replies":[%s]}' "$1" "$2" "$3" "$4"; }
{   # one English path (prompt -> answer A -> follow-up -> answer), one ending on a German reply
    printf '{"message_tree_id":"t1","tree_state":"ready_for_export","prompt":'
    m 'Q?' prompter en "$(m 'A1' assistant en "$(m 'More\nplease' prompter en "$(m 'A2' assistant en '')")"),$(m 'B1' assistant en "$(m 'Nein' prompter de "$(m 'x' assistant de '')")")"
    printf '}\n'
} > "$work/trees.jsonl"
./dialogprep oasst "$work/oa" test_frac=0 < "$work/trees.jsonl" 2> /dev/null
test "$(cat "$work/oa_train.bin")" = "$(printf '\036\002Q?\004\003A1\004\002More\nplease\004\003A2\004')"
echo 'PASS CLI: dense resume within FP32 tolerance, evaluation tails, unchanged checkpoints, invalid input rejection, MLA resume, trace resume, document resets, gradcheck, dialog training, init=, chat and dialogprep'
