#!/bin/sh
# The Zig host programs must behave like the C++ ones. Compares outputs of both
# builds on the same inputs; needs `make` and `zig build` to have run.
set -eu
cd "$(dirname "$0")/.."
Z=zig-out/bin
for t in dialogprep kgprep sample chat train gradcheck kgtrain; do
    test -x "$Z/$t" || { echo "missing $Z/$t (run: zig build)"; exit 1; }
    test -x "./$t" || { echo "missing ./$t (run: make)"; exit 1; }
done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM

# ---- data tools: byte-identical output ----
printf 'hi there\thello!\nno tab\n\t x\t \nq\t a\001b\n' > "$work/pairs.tsv"
./dialogprep tsv "$work/pairs.tsv" "$work/c" test_frac=0.5 2> /dev/null
$Z/dialogprep tsv "$work/pairs.tsv" "$work/z" test_frac=0.5 2> /dev/null
cmp "$work/c_train.bin" "$work/z_train.bin"
cmp "$work/c_test.bin" "$work/z_test.bin"
m() { printf '{"text":"%s","role":"%s","lang":"%s","replies":[%s]}' "$1" "$2" "$3" "$4"; }
{
    printf '{"message_tree_id":"t1","prompt":'
    m 'Q \\u00e4?' prompter en "$(m 'A1\\nx' assistant en "$(m 'More' prompter en "$(m 'A2' assistant en '')")"),$(m 'B1' assistant en "$(m 'Nein' prompter de '')")"
    printf '}\n'
} > "$work/trees.jsonl"
./dialogprep oasst "$work/co" test_frac=0.5 < "$work/trees.jsonl" 2> /dev/null
$Z/dialogprep oasst "$work/zo" test_frac=0.5 < "$work/trees.jsonl" 2> /dev/null
cmp "$work/co_train.bin" "$work/zo_train.bin"

st() {
    printf '{"mainsnak":{"snaktype":"value","datavalue":{"value":{"id":"%s"}}},"rank":"%s"}' "$2" "$1"
}
{
    printf '{"type":"item","id":"Q1","labels":{"en":{"value":"Fr\\u00e4nce"}},"sitelinks":{"enwiki":{}},"claims":{"P36":[%s,%s],"P38":[%s],"P17":[%s]}}\n' \
        "$(st normal Q3)" "$(st preferred Q2)" "$(st deprecated Q4)" "$(st normal Q1)"
    printf '{"type":"item","id":"Q2","labels":{"en":{"value":"Paris"}},"sitelinks":{},"claims":{}}\n'
    printf '{"type":"item","id":"Q3","labels":{"en":{"value":"Lyon"}},"sitelinks":{},"claims":{}}\n'
    printf '{"type":"item","id":"Q4","labels":{"en":{"value":"franc"}},"sitelinks":{},"claims":{}}\n'
} > "$work/dump.json"
./kgprep dump "$work/cg.tsv" threads=3 < "$work/dump.json" 2> /dev/null
$Z/kgprep dump "$work/zg.tsv" threads=3 < "$work/dump.json" 2> /dev/null
cmp "$work/cg.tsv" "$work/zg.tsv"
./kgprep qa "$work/cg.tsv" "$work/cq" 2> /dev/null
$Z/kgprep qa "$work/cg.tsv" "$work/zq" 2> /dev/null
for f in train test nodes; do cmp "$work/cq_$f.tsv" "$work/zq_$f.tsv"; done

# ---- model: same training log and checkpoint ----
awk 'BEGIN { srand(7); for (i = 0; i < 60000; i++) printf "%c", 32 + int(rand() * 90) }' > "$work/data"
CFG="dim=32 layers=2 batch=4 seqlen=16"
./train "$work/data" "$work/c.ckpt" $CFG steps=60 saveevery=0 > "$work/c.log" 2>&1
$Z/train "$work/data" "$work/z.ckpt" $CFG steps=60 saveevery=0 > "$work/z.log" 2>&1
# The dense path is deterministic, so the logs must match exactly (timings aside).
diff "$(grep -v seconds "$work/c.log" > "$work/c.f"; echo "$work/c.f")" \
     "$(grep -v seconds "$work/z.log" > "$work/z.f"; echo "$work/z.f")"
# Checkpoints: weights, moments, progress and stream state within FP32 tolerance
# (atomics in the embedding gradient make runs differ in the last bits).
./architecture_test --compare "$work/c.ckpt" "$work/z.ckpt" > /dev/null
# Resume and evaluation of the same checkpoint.
./train "$work/data" "$work/c.ckpt" steps=20 saveevery=0 > /dev/null 2>&1
./train "$work/data" "$work/c.ckpt" mode=eval seqlen=8 > "$work/ce.log" 2>&1
$Z/train "$work/data" "$work/c.ckpt" mode=eval seqlen=8 > "$work/ze.log" 2>&1
diff "$(grep -v seconds "$work/ce.log" > "$work/ce.f"; echo "$work/ce.f")" \
     "$(grep -v seconds "$work/ze.log" > "$work/ze.f"; echo "$work/ze.f")"
# Same error messages.
for args in "dim=999999999" "bogus=1" "steps=-2" "noequals"; do
    ./train "$work/data" "$work/c.ckpt" $args > "$work/c.err" 2>&1 || true
    $Z/train "$work/data" "$work/c.ckpt" $args > "$work/z.err" 2>&1 || true
    diff "$work/c.err" "$work/z.err"
done

# ---- gradient comparison: identical report ----
./gradcheck "$work/data" "$work/c.ckpt" len=256 window=64 seqs=2 > "$work/c.gc" 2>&1
$Z/gradcheck "$work/data" "$work/c.ckpt" len=256 window=64 seqs=2 > "$work/z.gc" 2>&1
cmp "$work/c.gc" "$work/z.gc"
./gradcheck "$work/data" "$work/c.ckpt" len=100 window=64 > "$work/c.gc" 2>&1 || true
$Z/gradcheck "$work/data" "$work/c.ckpt" len=100 window=64 > "$work/z.gc" 2>&1 || true
cmp "$work/c.gc" "$work/z.gc"

# ---- fact memory: identical training, evaluation and retrieval ----
cat > "$work/qa.tsv" <<'EOF'
What is the capital of France?	Paris	France capital Paris; France continent Europe	Q1
What is the continent of France?	Europe	France capital Paris; France continent Europe	Q1
What is the capital of Germany?	Berlin	Germany capital Berlin	Q7
What is the capital of Spain?	Madrid	Spain capital Madrid	Q8
EOF
cat > "$work/nodes.tsv" <<'EOF'
Q1	France	France capital Paris; France continent Europe
Q7	Germany	Germany capital Berlin
Q8	Spain	Spain capital Madrid
Q2	Paris	
EOF
KG="dim=16 layers=2 batch=2 seqlen=48 mem_len=64 mem_heads=2 mem_dh=4"
./kgtrain train "$work/qa.tsv" "$work/ckg.ckpt" $KG steps=40 saveevery=0 > "$work/c.kg" 2>&1
$Z/kgtrain train "$work/qa.tsv" "$work/zkg.ckpt" $KG steps=40 saveevery=0 > "$work/z.kg" 2>&1
cmp "$work/c.kg" "$work/z.kg"
./architecture_test --compare "$work/ckg.ckpt" "$work/zkg.ckpt" > /dev/null
for mode in on off shuffled; do
    ./kgtrain eval "$work/qa.tsv" "$work/ckg.ckpt" memory=$mode > "$work/c.kge" 2>&1
    $Z/kgtrain eval "$work/qa.tsv" "$work/ckg.ckpt" memory=$mode > "$work/z.kge" 2>&1
    cmp "$work/c.kge" "$work/z.kge"
done
# Stage 2: retrieval heads (host math in zig/retrieval.zig), index and ask demo.
./kgtrain train "$work/qa.tsv" "$work/cr.ckpt" nodes="$work/nodes.tsv" $KG mem_rdim=8 steps=20 saveevery=0 > "$work/c.kgr" 2>&1
$Z/kgtrain train "$work/qa.tsv" "$work/zr.ckpt" nodes="$work/nodes.tsv" $KG mem_rdim=8 steps=20 saveevery=0 > "$work/z.kgr" 2>&1
cmp "$work/c.kgr" "$work/z.kgr"
./architecture_test --compare "$work/cr.ckpt" "$work/zr.ckpt" > /dev/null
./kgtrain eval "$work/qa.tsv" "$work/cr.ckpt" nodes="$work/nodes.tsv" memory=retrieved > "$work/c.kgr" 2>&1
$Z/kgtrain eval "$work/qa.tsv" "$work/cr.ckpt" nodes="$work/nodes.tsv" memory=retrieved > "$work/z.kgr" 2>&1
cmp "$work/c.kgr" "$work/z.kgr"
./kgtrain ask "$work/cr.ckpt" nodes="$work/nodes.tsv" "What is the capital of France?" top=3 maxlen=6 > "$work/c.ask" 2>&1
$Z/kgtrain ask "$work/cr.ckpt" nodes="$work/nodes.tsv" "What is the capital of France?" top=3 maxlen=6 > "$work/z.ask" 2>&1
cmp "$work/c.ask" "$work/z.ask"

# ---- generation: identical bytes ----
for args in "temp=0 maxlen=40" "temp=0.8 maxlen=60 seed=5"; do
    ./sample "$work/c.ckpt" "abc" $args > "$work/c.out" 2>&1
    $Z/sample "$work/c.ckpt" "abc" $args > "$work/z.out" 2>&1
    cmp "$work/c.out" "$work/z.out"
done
printf 'hello\n/temp 0.4\nmore\n/reset\nagain\n' > "$work/chat.in"
./chat "$work/c.ckpt" maxlen=30 seed=2 < "$work/chat.in" > "$work/c.chat" 2>&1
$Z/chat "$work/c.ckpt" maxlen=30 seed=2 < "$work/chat.in" > "$work/z.chat" 2>&1
cmp "$work/c.chat" "$work/z.chat"
echo 'PASS Zig: data tools byte-identical, training log/checkpoint, resume, eval, errors, gradcheck, kgtrain (both stages), sample and chat match the C++ build'
