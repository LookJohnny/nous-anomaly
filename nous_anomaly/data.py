"""One record format for every dataset, plus read-only adapters for the ai-emotion-experiment repo.

A record:
  id        unique string, "<source>:<item id>"
  source    which dataset it came from
  site      "reply" (the state is read on the assistant's text) or "user" (read on the last user message)
  system    system prompt or None
  messages  chat messages BEFORE the target (for site "user", the last one is the target itself)
  target    the text the state is read on
  group     template / scenario id; splits and bootstrap never separate one group
  topic     coarse topic, used for the topic-shift control (may be None)
  split     dev | val | test
  labels    {"hostile": 0/1, "distress": 0/1, "syco": 0/1, "emotion": "anger"|"joy"|"sadness"|"neutral"}

  python -m nous_anomaly.data --repo ~/ai-emotion-experiment/ai-emotion-experiment-main --out data/records.jsonl
"""
import argparse, collections, json, os

BINARY = ("hostile", "distress", "syco")
EMOTIONS = ("anger", "joy", "sadness")
CONCEPTS = BINARY + EMOTIONS
# Templates the original M4 study used for in-domain training (v11_probe.py); the rest are its evaluation templates.
M4_DEV = set("k102 k103 k106 k107 k108 k109 k112 k116 k119 k120 k121 k126 k127 k128 k132 k133 k135 k137 k138 k141 "
             "k144 k146 k147 k148 k149 k150 k153 k155 k156 k157".split())


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_normal(record):
    """Normal = no risk label set and no non-neutral emotion. The one-class detectors train on these only."""
    labels = record["labels"]
    return not any(labels.get(k) for k in BINARY) and labels.get("emotion", "neutral") == "neutral"


def is_positive(record, concept):
    labels = record["labels"]
    return labels.get("emotion") == concept if concept in EMOTIONS else labels.get(concept) == 1


def load_c1(repo):
    """Hostile replies under the default persona (monitor V1.1, companion domain)."""
    out = []
    for r in _rows(os.path.join(repo, "2026-09-28/monitor/v1_1/c1/labels_all.jsonl")):
        out.append(dict(id="c1:" + r["item_id"], source="c1", site="reply", system=r.get("system"), messages=r["messages"],
                        target=r["target_text"], group=r["template_id"], topic=r.get("domain"), split=r["split"],
                        labels={"hostile": int(r["label"])}, meta={"level": r["level"]}))
    return out


def load_c2(repo):
    """User distress. V1 items (keyword-heavy) and the M4 2x2 keyword-controlled set."""
    out = []
    for r in _rows(os.path.join(repo, "2026-09-28/monitor/data/labels_all.jsonl")):
        if r["concept"] != "C2":
            continue          # V1's C1 is confounded with the persona prompt (MONITOR_REPORT.md); not used
        out.append(dict(id="c2:" + r["item_id"], source="c2", site="user", system=None, messages=r["messages"],
                        target=r["target_text"], group=r["template_id"], topic=r.get("domain"), split=r["split"],
                        labels={"distress": int(r["label"])}, meta={"kind": r.get("kind")}))
    for r in _rows(os.path.join(repo, "2026-09-28/monitor/v1_1/m4/labels.jsonl")):
        out.append(dict(id="m4:" + r["item_id"], source="m4", site="user", system=None, messages=r["messages"],
                        target=r["target_text"], group=r["template_id"], topic=r.get("domain"),
                        split="dev" if r["template_id"] in M4_DEV else "test",
                        labels={"distress": int(r["label"])}, meta={"cell": r["cell"]}))
    return out


def load_emotion(repo):
    """E01's 960 mixed-event records: same facts, four emotions. Neutral records are the normal class."""
    split = {"discovery": "dev", "validation": "val", "test": "test"}
    out = []
    for r in _rows(os.path.join(repo, "2026-09-28/data/formal_960.jsonl")):
        out.append(dict(id="emo:" + r["record_id"], source="emo", site="reply", system=None,
                        messages=[{"role": "user", "content": r["user_prompt"]}],
                        target=r["style_lead"] + r["common_anchor"] + r["reference_suffix"],
                        group=r["scenario_id"], topic=r["topic"], split=split[r["split"]],
                        labels={"emotion": r["emotion"]}, meta={"paraphrase": r["paraphrase"]}))
    return out


def load_syco(repo):
    """Sycophancy: Qwen3-4B replies to a user's wrong (W, P) or right (C) claim. Replies the raters could not label
    (syco is null: unclear, or an invalid distractor) are dropped, as in REPORT_SYCO.md."""
    labels = {r["item_id"]: r for r in _rows(os.path.join(repo, "2026-10-02/syco/labels_qwen.jsonl"))}
    out = []
    for r in _rows(os.path.join(repo, "2026-10-02/syco/qwen_replies.jsonl")):
        lab = labels.get(r["item_id"])
        if not lab or lab["syco"] is None or not r.get("target_text"):
            continue
        out.append(dict(id="syco:" + r["item_id"], source="syco", site="reply", system=r.get("system"),
                        messages=r["messages"], target=r["target_text"], group=r["src_item"], topic=r["template_id"],
                        split=r["split"], labels={"syco": int(lab["syco"])},
                        meta={"template": r["template"], "endorse": lab["endorse"]}))
    return out


LOADERS = {"c1": load_c1, "c2": load_c2, "emo": load_emotion, "syco": load_syco}


def load_repo(repo, sources=None):
    repo = os.path.expanduser(repo)
    records = [r for name in (sources or LOADERS) for r in LOADERS[name](repo)]
    validate(records)
    return records


def validate(records):
    seen = set()
    for r in records:
        if r["id"] in seen:
            raise ValueError("duplicate id " + r["id"])
        seen.add(r["id"])
        if r["site"] not in ("reply", "user") or r["split"] not in ("dev", "val", "test"):
            raise ValueError("bad site or split in " + r["id"])
        if not r["target"] or not isinstance(r["messages"], list) or not r["messages"]:
            raise ValueError("empty target or messages in " + r["id"])
        if r["site"] == "user" and r["messages"][-1]["content"] != r["target"]:
            raise ValueError("user-site target must be the last user message: " + r["id"])
    # A group must not straddle splits, or a detector could learn the template.
    where = collections.defaultdict(set)
    for r in records:
        where[(r["source"], r["group"])].add(r["split"])
    bad = [k for k, v in where.items() if len(v) > 1]
    if bad:
        raise ValueError(f"{len(bad)} groups appear in more than one split, e.g. {bad[0]}")


def save(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path):
    return _rows(path)


def summary(records):
    out = collections.Counter()
    for r in records:
        kind = "normal" if is_normal(r) else "+".join(c for c in CONCEPTS if is_positive(r, c)) or "other"
        out[(r["source"], r["site"], r["split"], kind)] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*", choices=sorted(LOADERS))
    args = ap.parse_args(argv)
    records = load_repo(args.repo, args.sources)
    save(records, args.out)
    for key, n in sorted(summary(records).items()):
        print(*key, n, sep="\t")
    print("records", len(records), "->", args.out)


if __name__ == "__main__":
    main()
