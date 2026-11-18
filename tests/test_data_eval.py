import collections, json, os
import numpy as np
import pytest

from nous_anomaly import data, evaluate
from synth import D, World

REPO = os.path.expanduser(os.environ.get("NOUS_EMOTION_REPO", "~/ai-emotion-experiment/ai-emotion-experiment-main"))


class FakeFeatures:
    """Stands in for extract.Features: two layers, the risky signal exists only at layer 2."""
    layers = [1, 2]
    meta = {"model": "synthetic"}

    def __init__(self, vectors):
        self.vectors = vectors

    def __contains__(self, record_id):
        return record_id in self.vectors

    def get(self, ids, pool, layer):
        return np.stack([self.vectors[i][layer] for i in ids])


def _records_and_features(seed=0):
    w = World(seed); records, vectors = [], {}
    def add(source, site, split, group, labels, shift):
        topic = topics.setdefault((source, group), w.topic_mean())
        rid = f"{source}:{len(records)}"
        msgs = [{"role": "user", "content": "问题" if site == "reply" else "文本" + rid}]
        records.append(dict(id=rid, source=source, site=site, system=None, messages=msgs,
                            target=("好的" if not shift else "滚开") + rid if site == "reply" else "文本" + rid,
                            group=group, topic=None, split=split, labels=labels))
        base = w.sample(1, topic)[0]
        vectors[rid] = {1: base, 2: base + (8.0 * w.risk if shift else 0)}
    topics = {}
    for split, n_groups in (("dev", 40), ("val", 15), ("test", 15)):
        for g in range(n_groups):
            for k in range(10):
                add("c1", "reply", split, f"{split}{g}", {"hostile": int(k < 2)}, k < 2)
                add("emo", "reply", split, f"{split}s{g}", {"emotion": "anger" if k < 3 else "neutral"}, k < 3)
                add("c2", "user", split, f"{split}u{g}", {"distress": int(k < 2)}, k < 2)
    return records, FakeFeatures(vectors)


def test_metrics():
    assert evaluate.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert np.isnan(evaluate.auroc([0, 0], [0.1, 0.2]))
    y = [0] * 100 + [1] * 10; s = list(range(100)) + [200] * 5 + [0] * 5
    assert evaluate.recall_at_fpr(y, s, 0.05) == 0.5
    lo, hi = evaluate.bootstrap([0, 1] * 50, np.tile([0.0, 1.0], 50) + np.random.default_rng(0).normal(0, 0.4, 100),
                                [i // 4 for i in range(100)], n=200)
    assert 0.8 < lo <= hi <= 1.0


def test_validate_rejects_a_group_in_two_splits_and_duplicate_ids():
    r = dict(id="a:1", source="a", site="reply", system=None, messages=[{"role": "user", "content": "q"}], target="t",
             group="g", topic=None, split="dev", labels={})
    with pytest.raises(ValueError):
        data.validate([r, dict(r)])
    with pytest.raises(ValueError):
        data.validate([r, dict(r, id="a:2", split="test")])
    with pytest.raises(ValueError):
        data.validate([dict(r, site="user")])                     # user-site target must be the last user message
    data.validate([r, dict(r, id="a:2")])


def test_sweep_freeze_test_roundtrip(tmp_path):
    records, feats = _records_and_features()
    data.validate(records)
    result = evaluate.sweep(records, feats)
    assert result["split"] == "val" and len(result["runs"]) == (3 + 1) * 2
    frozen = evaluate.freeze(result)
    # The signal was planted at layer 2 only, so every detector must freeze on layer 2.
    assert {c["layer"] for c in frozen["chosen"].values()} == {2}
    assert set(frozen["chosen"]) == {f"{site}/{d}" for site in ("reply", "user") for d in evaluate.DEFAULT_DETECTORS}
    test = evaluate.run_frozen(records, feats, frozen, n_boot=100)
    for run in test["runs"]:
        (name, res), = run["detectors"].items()
        assert run["split"] == "test"
        if name in ("mahalanobis", "content_free"):
            assert res["fpr_new_normal"] < 0.2 and abs(res["shift_auroc"] - 0.5) < 0.15
            for m in res["concepts"].values():
                assert m["auroc"] > 0.9 and m["ci"][0] <= m["auroc"] <= m["ci"][1]
        assert set(res["concepts"]) == ({"hostile", "anger"} if run["site"] == "reply" else {"distress"})
        assert all(v["probe_auroc"] > 0.9 for v in run["references"].values())
    md = evaluate.table(test)
    assert "content_free" in md and "probe (supervised)" in md
    with pytest.raises(ValueError):
        evaluate.freeze(dict(result, split="test"))


def test_layer_without_signal_scores_at_chance():
    records, feats = _records_and_features()
    run = evaluate.evaluate(records, feats, site="reply", pool="mean", layer=1, n_boot=0, references=False)
    assert abs(run["detectors"]["content_free"]["concepts"]["hostile"]["auroc"] - 0.5) < 0.15


def test_cli_refuses_to_open_test_twice(tmp_path, monkeypatch):
    records, feats = _records_and_features()
    rec = tmp_path / "records.jsonl"; data.save(records, rec)
    from nous_anomaly import extract
    monkeypatch.setattr(extract, "Features", lambda path: feats)
    sweep, frz, out = tmp_path / "sweep.json", tmp_path / "FREEZE.json", tmp_path / "test.json"
    evaluate.main(["sweep", "--records", str(rec), "--feats", "x", "--out", str(sweep), "--detectors", "content_free"])
    evaluate.main(["freeze", "--sweep", str(sweep), "--out", str(frz)])
    evaluate.main(["test", "--records", str(rec), "--feats", "x", "--freeze", str(frz), "--out", str(out)])
    assert json.load(open(out))["split"] == "test"
    with pytest.raises(SystemExit):
        evaluate.main(["test", "--records", str(rec), "--feats", "x", "--freeze", str(frz), "--out", str(out)])
    with pytest.raises(SystemExit):
        evaluate.main(["freeze", "--sweep", str(sweep), "--out", str(frz)])


@pytest.mark.skipif(not os.path.isdir(REPO), reason="ai-emotion-experiment repo not present")
def test_real_repo_adapters():
    records = data.load_repo(REPO)
    by_source = collections.Counter(r["source"] for r in records)
    assert by_source["c1"] == 800 and by_source["emo"] == 960 and by_source["m4"] == 480
    assert by_source["c2"] == 423 and by_source["syco"] == 1497           # 1560 minus 63 unclear
    hostile = [r for r in records if r["source"] == "c1" and r["split"] == "test" and r["labels"]["hostile"]]
    assert len(hostile) == 38                                              # REPORT_v11.md: test 240 items / 38 positive
    syco = [r for r in records if r["source"] == "syco" and r["labels"]["syco"]]
    assert len(syco) == 287                                                # REPORT_SYCO.md: W 163 + P 124
    emo = [r for r in records if r["source"] == "emo"]
    assert collections.Counter(r["labels"]["emotion"] for r in emo) == {e: 240 for e in ("joy", "sadness", "anger", "neutral")}
    assert all(r["target"].startswith(r["target"][:1]) and "我的看法是：" in r["target"] for r in emo)
    assert sum(data.is_normal(r) for r in records) > 1500


def test_fewshot_curve_rises_with_examples_and_reports_all_methods():
    records, feats = _records_and_features()
    result = evaluate.fewshot(records, feats, draws=5)
    assert [(r["site"], r["pool"], r["layer"]) for r in result["runs"]] == [("reply", "mean", 2), ("reply", "pre", 2), ("user", "mean", 2)]
    reply = result["runs"][0]["concepts"]
    assert set(reply) == {"c1/hostile", "emo/anger"} and set(result["runs"][2]["concepts"]) == {"c2/distress"}
    curve = reply["c1/hostile"]["curve"]
    assert list(curve) == ["1", "2", "5", "10", "20", "all"] and curve["all"]["n"] == 80
    assert curve["5"]["few_shot"]["mean"] > 0.95 and curve["all"]["few_shot"]["sd"] == 0.0
    assert set(curve["5"]) == {"n", "few_shot", "probe", "text"}
    assert "c1/hostile" in evaluate.fewshot_table(result) and "k=all" in evaluate.fewshot_table(result, "text")
