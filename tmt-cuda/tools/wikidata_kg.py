#!/usr/bin/env python3
"""Fetch a Wikidata subset through the public API into a graph TSV for kgprep.

Network-bound helper (standard library only); everything compute-heavy lives in
tools/kgprep.cpp: parsing the full JSON dump (`kgprep dump`) and building the
QA data (`kgprep qa`). Seeds are search filters, e.g. "P31=Q6256" (country).

Graph TSV:   N <tab> QID <tab> label       (node)
             F <tab> QID <tab> PID <tab> QID   (fact: subject, property, object)
Keep PROPS in sync with tools/kgprep.cpp.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "tmt-cuda-kg/0.1 (research prototype; https://github.com/SimonVitzethum/test-model-thing-CUDA)"

# Item-valued properties used for facts: short name in memory, question template.
PROPS = {
    "P36": ("capital", "What is the capital of {s}?"),
    "P30": ("continent", "On which continent is {s}?"),
    "P38": ("currency", "What is the currency of {s}?"),
    "P37": ("official language", "What is the official language of {s}?"),
    "P17": ("country", "In which country is {s}?"),
    "P19": ("place of birth", "Where was {s} born?"),
    "P20": ("place of death", "Where did {s} die?"),
    "P27": ("citizenship", "What is the citizenship of {s}?"),
    "P106": ("occupation", "What was the occupation of {s}?"),
    "P50": ("author", "Who wrote {s}?"),
    "P136": ("genre", "What is the genre of {s}?"),
    "P495": ("country of origin", "Where does {s} come from?"),
    "P186": ("made from", "What is {s} made of?"),
    "P57": ("director", "Who directed {s}?"),
    "P1412": ("language spoken", "Which language did {s} speak?"),
}
MAX_LABEL = 60


def clean(label):
    label = " ".join(label.replace("\t", " ").replace("\n", " ").split())
    return label if 0 < len(label.encode("utf-8")) <= MAX_LABEL else None


def claims_of(entity):
    """Item-valued claims for PROPS; preferred rank wins over normal, deprecated dropped."""
    out = []
    for pid in PROPS:
        statements = [s for s in entity.get("claims", {}).get(pid, []) if s.get("rank") != "deprecated"]
        if any(s.get("rank") == "preferred" for s in statements):
            statements = [s for s in statements if s.get("rank") == "preferred"]
        for s in statements:
            snak = s.get("mainsnak", {})
            value = snak.get("datavalue", {}).get("value")
            if snak.get("snaktype") == "value" and isinstance(value, dict) and "id" in value:
                out.append((pid, value["id"]))
    return out


def en_label(entity):
    return clean(entity.get("labels", {}).get("en", {}).get("value", ""))


# ---------------------------------------------------------------- API mode
def api_get(params, retries=5):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            if "error" in data:
                raise RuntimeError(data["error"].get("info", "API error"))
            return data
        except Exception as e:  # network hiccup or server lag: back off politely
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry after {e}", file=sys.stderr)


def search_seed(seed, limit):
    """QIDs matching a haswbstatement filter, e.g. 'P31=Q6256'."""
    ids, offset = [], 0
    while len(ids) < limit and offset < 10000:
        data = api_get({"action": "query", "list": "search", "srsearch": f"haswbstatement:{seed}",
                        "srlimit": min(50, limit - len(ids)), "sroffset": offset, "srnamespace": 0})
        hits = data.get("query", {}).get("search", [])
        ids += [h["title"] for h in hits if h["title"].startswith("Q")]
        if "continue" not in data:
            break
        offset = data["continue"]["sroffset"]
        time.sleep(0.2)
    return ids[:limit]


def get_entities(ids, props):
    out = {}
    for i in range(0, len(ids), 50):
        data = api_get({"action": "wbgetentities", "ids": "|".join(ids[i:i + 50]),
                        "props": props, "languages": "en"})
        out.update({k: v for k, v in data.get("entities", {}).items() if "missing" not in v})
        time.sleep(0.2)
    return out


def cmd_fetch(args):
    subjects = []
    for seed in args.seed:
        found = search_seed(seed, args.per_seed)
        print(f"seed {seed}: {len(found)} entities", file=sys.stderr)
        subjects += found
    subjects = list(dict.fromkeys(subjects))
    labels, facts = {}, []
    entities = get_entities(subjects, "labels|claims")
    for qid, e in entities.items():
        label = en_label(e)
        if not label:
            continue
        labels[qid] = label
        facts += [(qid, pid, obj) for pid, obj in claims_of(e)]
    missing = sorted({o for _, _, o in facts} - labels.keys())
    for qid, e in get_entities(missing, "labels").items():
        label = en_label(e)
        if label:
            labels[qid] = label
    write_graph(args.out, labels, facts)


def write_graph(path, labels, facts):
    facts = [f for f in dict.fromkeys(facts) if f[0] in labels and f[2] in labels and f[0] != f[2]]
    used = {f[0] for f in facts} | {f[2] for f in facts}
    with open(path, "w", encoding="utf-8") as out:
        for qid in sorted(used, key=lambda q: int(q[1:])):
            out.write(f"N\t{qid}\t{labels[qid]}\n")
        for s, p, o in facts:
            out.write(f"F\t{s}\t{p}\t{o}\n")
    print(f"wrote {path}: {len(used):,} nodes, {len(facts):,} facts", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", action="append", required=True, help='search filter, e.g. "P31=Q6256"; repeatable')
    ap.add_argument("--per-seed", type=int, default=500)
    ap.add_argument("--out", required=True)
    cmd_fetch(ap.parse_args())


if __name__ == "__main__":
    main()
