#!/bin/sh
# Knowledge-graph pipeline without network: synthetic Wikidata dump -> kgprep
# dump/qa -> kgtrain train/eval smoke test.
set -eu
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
st() { # rank, object -> statement
    printf '{"mainsnak":{"snaktype":"value","property":"P","datavalue":{"value":{"entity-type":"item","numeric-id":1,"id":"%s"},"type":"wikibase-entityid"}},"type":"statement","rank":"%s"}' "$2" "$1"
}
{
    echo '['
    # Country with preferred capital (the normal-rank one must be dropped), a deprecated
    # currency, a self-reference and an escaped label.
    printf '{"type":"item","id":"Q1","labels":{"en":{"language":"en","value":"Fr\\u00e4nce \\"X\\""}},"sitelinks":{"enwiki":{"site":"enwiki","title":"France"}},"claims":{"P36":[%s,%s],"P38":[%s],"P30":[%s],"P17":[%s]}},\n' \
        "$(st normal Q3)" "$(st preferred Q2)" "$(st deprecated Q4)" "$(st normal Q5)" "$(st normal Q1)"
    printf '{"type":"item","id":"Q2","labels":{"en":{"language":"en","value":"Paris"}},"sitelinks":{},"claims":{}},\n'
    printf '{"type":"item","id":"Q3","labels":{"en":{"language":"en","value":"Lyon"}},"sitelinks":{},"claims":{}},\n'
    printf '{"type":"item","id":"Q4","labels":{"en":{"language":"en","value":"franc"}},"sitelinks":{},"claims":{}},\n'
    printf '{"type":"item","id":"Q5","labels":{"en":{"language":"en","value":"Europe"}},"sitelinks":{},"claims":{}},\n'
    # No English Wikipedia article: its facts are ignored, its label is still usable.
    printf '{"type":"item","id":"Q6","labels":{"en":{"language":"en","value":"Nowhere"}},"sitelinks":{"dewiki":{}},"claims":{"P36":[%s]}},\n' "$(st normal Q2)"
    printf '{"type":"property","id":"P36","labels":{"en":{"language":"en","value":"capital"}}}\n'
    echo ']'
} > "$work/dump.json"
./kgprep dump "$work/graph.tsv" threads=2 < "$work/dump.json" 2> "$work/dump.log"
grep -q '^N	Q1	Fränce "X"$' "$work/graph.tsv"
grep -q '^F	Q1	P36	Q2$' "$work/graph.tsv"
! grep -q 'P36	Q3' "$work/graph.tsv"       # preferred rank wins
! grep -q 'P38' "$work/graph.tsv"            # deprecated dropped
! grep -q 'F	Q1	P17	Q1' "$work/graph.tsv"   # self-reference dropped
! grep -q 'F	Q6' "$work/graph.tsv"          # no enwiki article
test "$(grep -c '^F' "$work/graph.tsv")" -eq 2
./kgprep qa "$work/graph.tsv" "$work/qa" test_frac=0 2> "$work/qa.log"
grep -q '^What is the capital of Fränce "X"?	Paris	' "$work/qa_train.tsv"
test "$(wc -l < "$work/qa_train.tsv")" -eq 2
test ! -s "$work/qa_test.tsv"
# Training/evaluation smoke test on the generated rows.
./kgtrain train "$work/qa_train.tsv" "$work/kg.ckpt" dim=16 layers=2 batch=2 seqlen=48 mem_len=64 mem_heads=2 mem_dh=4 steps=30 saveevery=0 > "$work/train.log"
./kgtrain train "$work/qa_train.tsv" "$work/kg.ckpt" steps=10 saveevery=0 >> "$work/train.log"
for mode in on off shuffled; do
    ./kgtrain eval "$work/qa_train.tsv" "$work/kg.ckpt" memory=$mode > "$work/eval-$mode.log"
    grep -q '"examples":2,' "$work/eval-$mode.log"
done
./sample "$work/kg.ckpt" "What is" maxlen=8 > "$work/sample.log"
echo 'PASS KG: dump parsing (ranks, enwiki filter, escapes, self-references), QA split, kgtrain resume/eval, sampler without memory'
