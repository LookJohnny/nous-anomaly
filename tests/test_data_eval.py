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


def _with_pairs(records, feats, w_seed=0):
    """Add generated contrast pairs for "hostile": same risky direction as the real data, different topics."""
    w = World(w_seed); rng = np.random.default_rng(1)
    for n in range(30):
        topic = w.topic_mean()
        for label in (0, 1):
            rid = f"syn:hostile:{n}:{label}"
            records.append(dict(id=rid, source="syn", site="reply", system=None, messages=[{"role": "user", "content": "问题"}],
                                target=("滚开" if label else "好的") + rid, group=f"hostile:{n}", topic=None, split="syn",
                                labels={"hostile": 1} if label else {}, meta={"concept": "hostile", "context": "x"}))
            base = topic + rng.normal(size=len(topic))
            feats.vectors[rid] = {1: base, 2: base + (8.0 * w.risk if label else 0)}
    return records, feats


def test_generated_pairs_define_a_direction_that_works_on_real_records():
    records, feats = _with_pairs(*_records_and_features())
    data.validate(records)
    before = evaluate.fewshot(*_records_and_features(), draws=2, ks=(5,))
    after = evaluate.fewshot(records, feats, draws=2, ks=(5,))
    assert before["runs"][0] == after["runs"][0]                # "syn" records never leak into normal or evaluation data
    result = evaluate.contrast(records, feats, draws=3)
    run = result["runs"][0]
    assert run["pool"] == "mean" and set(run["concepts"]) == {"c1/hostile"}     # anger has no pairs
    curve = run["concepts"]["c1/hostile"]["curve"]
    assert curve["all"]["n"] == 30 and curve["5"]["few_shot"] > 0.95 and curve["all"]["text"] > 0.95


def test_fewshot_geometry_can_use_a_subset_of_the_normal_records():
    records, feats = _records_and_features()
    small = evaluate.fewshot(records, feats, draws=2, ks=(5,), max_normal=100)
    assert small["runs"][0]["n_normal"] == 100 and evaluate.fewshot(records, feats, draws=2, ks=(5,))["runs"][0]["n_normal"] > 100


def test_token_readout_finds_a_signal_carried_by_a_few_tokens():
    records, feats = _records_and_features()
    w = World(0); rng = np.random.default_rng(2); toks = {}
    for r in records:
        t = feats.vectors[r["id"]][1] + rng.normal(size=(20, 96))
        if data.is_positive(r, "hostile") or data.is_positive(r, "anger") or data.is_positive(r, "distress"):
            t[rng.choice(20, 2, replace=False)] += 12.0 * w.risk          # two of twenty tokens carry it
        toks[r["id"]] = t
    feats.tokens = lambda ids: [toks[i] for i in ids]
    feats.meta = {"token_layer": 2}
    result = evaluate.tokens(records, feats, draws=3)
    k = result["concepts"]["reply/c1/hostile"]["k"]
    assert set(result["concepts"]) == {"reply/c1/hostile", "reply/emo/anger", "user/c2/distress"}
    assert k["all"]["max"] > k["all"]["mean"] and k["all"]["max"] > 0.95 and set(k) == {"5", "all"}


def test_generated_records_are_valid_and_hide_the_instruction():
    from nous_anomaly import generate
    def fake(prompts, max_new_tokens=0):
        return ["1. 怎么煮米饭才好吃？\n2、推荐一道家常菜\n- 推荐一道家常菜\n好" if m[-1]["content"].startswith(("请写", "Write"))
                else "指令" if "数据构造任务" in (s or "") else "回复" for s, m in prompts]
    normal = generate.normal_records(fake, per_cell=10)
    assert len(normal) == 1000 and {r["split"] for r in normal} == {"dev", "val"} and all(data.is_normal(r) for r in normal)
    assert normal[0]["messages"][0]["content"] == "怎么煮米饭才好吃？"           # numbering stripped, duplicates dropped
    records, _ = _records_and_features()
    pairs = generate.contrast_records(fake, records, per_concept=6, concepts=("hostile", "anger"))   # 2 distinct dialogues
    data.validate(records + normal + pairs)
    assert len(pairs) == 8 and all(r["split"] == "syn" and "数据构造任务" not in (r["system"] or "") for r in pairs)
    assert [(r["target"], data.is_positive(r, "hostile")) for r in pairs[:2]] == [("回复", False), ("指令", True)]
    assert pairs[-1]["labels"] == {"emotion": "anger"} and {r["meta"]["context"].split(":")[0] for r in pairs} == {"c1", "emo"}


def test_drift_set_rating_and_sealed_evaluation(tmp_path):
    from nous_anomaly import generate, rate
    def fake(prompts, max_new_tokens=0, temperature=1, top_p=1):
        out = []
        for s, m in prompts:
            c = m[-1]["content"]
            if "|||" in c:
                out.append("\n".join(f"{i}. 我的订单{i}没到 ||| 已经等了{i}天 ||| 能帮我查一下吗" for i in range(1, 16)))
            elif c.startswith(rate.RUBRIC):
                out.append("3" if "滚" in c.split("【待评回复】")[-1] else "0。")
            else:
                out.append("滚" if "辱骂" in (s or "") or (temperature == 0.9 and "滚" in str(m) and len(out) % 3 == 0) else "好的")
        return out
    hx = generate.drift_records(fake)
    assert len(hx) == 1440 and all("数据构造任务" not in r["system"] for r in hx) and {r["split"] for r in hx} == {"test"}
    scores = rate.score(fake, hx)
    rated, stats = rate.combine(hx, [scores, dict(scores, **{hx[0]["id"]: None})])
    assert stats["dropped"] == 1 and stats["kappa"] == 1.0 and 0 < stats["positive"] < 900
    assert all(r["meta"]["level"] == "insulting" for r in rated if r["labels"]["hostile"])
    assert rate.sheet(rated, tmp_path / "sheet.csv") == 100 and rate.human(rated, tmp_path / "sheet.csv") == {"filled": 0}
    assert rate.kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5) and rate.parse("评分：2") == 2 and rate.parse("无") is None

    # Sealed evaluation on synthetic vectors: the signal sits on the risky direction of layer 2 for hostile replies.
    records, feats = _with_pairs(*_records_and_features())
    for r in records:
        r.setdefault("meta", {})["level"] = "insulting" if r["source"] == "c1" else "x"
    w = World(0); rng = np.random.default_rng(3)
    for r in rated[:400]:
        records.append(r); base = w.sample(1, w.topic_mean())[0]
        feats.vectors[r["id"]] = {1: base, 2: base + (8.0 * w.risk if r["labels"]["hostile"] else 0)}
    result = evaluate.sealed(records, feats, n_boot=50, draws=3)
    new, old = result["hostile"]["hx"], result["hostile"]["c1"]
    assert new["within_insulting"]["n_pos"] > 20 and new["overall"]["n_neg"] > new["within_insulting"]["n_neg"]
    m = new["within_insulting"]["methods"]
    assert m["syn_early"]["auroc"] > 0.95 and m["real5_mean"]["auroc"] > 0.95 and m["syn_early"]["ci"][0] > 0.9
    assert set(m) == {"syn_early", "syn_mean", "text_syn", "real5_mean", "real_all_mean", "probe_mean", "pre_all", "text_real"}
    assert "syn_early_minus_text_syn" in new["within_insulting"] and old["overall"]["n_pos"] == 30


def test_risk_set_and_prompt_level_forecast():
    from nous_anomaly import generate, rate
    def fake(prompts, max_new_tokens=0, temperature=1, top_p=1):
        out = []
        for s, m in prompts:
            c = m[-1]["content"]
            if "|||" in c:
                out.append("\n".join(f"订单{len(out)}-{i}没到 ||| 等了{i}天 ||| 帮我查一下" for i in range(12)))
            elif c.startswith("把下面这句"):
                out.append("蠢货，" + c.split("\n\n")[-1])
            elif c.startswith(rate.FORECAST):
                out.append("3" if "滚" in c else "0")
            else:
                out.append("滚" if "辱骂" in (s or "") else "好的")
        return out
    rk = generate.risk_records(fake, per_domain=4, samples=3, domains=["客服", "陪伴"], calls=1)
    assert len(rk) == 2 * 4 * 3 * 3 and {r["meta"]["level"] for r in rk} == {"insulting", "polite", "curt", "sarcastic", "rude_user"}
    rude = [r for r in rk if r["meta"]["level"] == "rude_user"]
    assert rude[0]["messages"][-1]["content"].startswith("蠢货，") and rude[0]["messages"][1]["content"] == "好的"
    judge = rate.forecast(fake, rk)
    assert len(judge) == 24 and set(judge.values()) == {0, 3} and judge[rk[0]["meta"]["prompt"]] == 3

    # Forecast evaluation on synthetic vectors: risky prompts carry the signal before the reply.
    records, feats = _with_pairs(*_records_and_features())
    feats.layers = [2]
    w = World(0); rng = np.random.default_rng(5); n = 0
    for r in records:                                   # in the old set, hostile and plain replies come from different prompts
        r.setdefault("meta", {})["level"] = "insulting" if r["labels"].get("hostile") else "polite"
    for g in range(60):
        for level in ("insulting", "polite"):
            risky = level == "insulting" and g % 3 == 0
            base = w.sample(1, w.topic_mean())[0] + (8.0 * w.risk if risky else 0)
            for k in range(4):
                rid = f"rk:{g}-{level}-s{k}"; n += 1
                records.append(dict(id=rid, source="rk", site="reply", system=None, messages=[{"role": "user", "content": f"问题{g}"}],
                                    target="回复", group=f"g{g}", topic=None, split="test", labels={"hostile": int(risky and k < 2)},
                                    meta={"level": level, "prompt": f"{g}-{level}"}))
                feats.vectors[rid] = {1: base, 2: base}
    result = evaluate.forecast(records, feats, {f"{g}-insulting": g % 2 for g in range(60)}, n_boot=50)
    pop = result["populations"]["insulting"]
    assert result["n_prompts"] == 120 and pop["n_prompts"] == 60 and pop["n_positive"] == 20 and pop["hostile_replies"] == 40
    assert pop["methods"]["pre_L2"]["auroc"] > 0.95 and pop["methods"]["pre_L2"]["spearman_with_rate"] > 0.6
    assert pop["methods"]["pre_L2"]["hostile_replies_flagged"] > 0.9 and pop["methods"]["prompt_text"]["auroc"] < 0.75
    assert pop["pre_L2_minus_best_text"]["ci"][0] > 0 and "llm_judge" in pop["methods"]
    assert result["populations"]["polite"]["methods"] == {}
