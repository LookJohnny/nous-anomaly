"""Evaluate detectors on extracted features.

The one-class detectors are fitted on NORMAL dev records only and never see a risky example. Each concept
(hostile, distress, syco, anger, joy, sadness) is then a test of whether the detector flags something it was never
shown. Two references are trained WITH labels on the same dev data: a linear probe on the activations and a
character n-gram classifier on the text.

Topic-shift control: a fifth of the normal dev groups is held out of fitting. The alarm threshold is the 95th
percentile of their scores. `fpr_new_normal` is the share of normal records in the evaluated split (new templates,
and for the emotion set new topics) above that threshold; it should stay near 0.05. `shift_auroc` compares the same
two sets and should stay near 0.5.

  python -m nous_anomaly.evaluate fewshot --records data/records.jsonl --feats feats/qwen3-4b --out out/fewshot_val.json
  python -m nous_anomaly.evaluate contrast --records ... --feats ... --out out/contrast_val.json   generated pairs -> real data
  python -m nous_anomaly.evaluate basis    --records ... --feats ... --out out/basis.npz          for extract --token-basis
  python -m nous_anomaly.evaluate tokens   --records ... --feats ... --out out/tokens_val.json    mean vs max over tokens
  python -m nous_anomaly.evaluate sealed   --records ... --feats A B --out out/sealed_test.json    PREREG_V2.md; runs once
  python -m nous_anomaly.evaluate sweep  --records data/records.jsonl --feats feats/qwen3-4b --out out/sweep_val.json
  python -m nous_anomaly.evaluate freeze --sweep out/sweep_val.json --out out/FREEZE.json
  python -m nous_anomaly.evaluate test   --records data/records.jsonl --feats feats/qwen3-4b --freeze out/FREEZE.json --out out/test.json
`test` refuses to run twice: the test split is opened once, with the frozen layer and pooling.
"""
import argparse, hashlib, json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import CONCEPTS, is_normal, is_positive, load
from .detectors import ONE_CLASS, LinearProbe

DEFAULT_DETECTORS = ("mahalanobis", "content_free", "knn", "isolation_forest")
SITE_POOLS = {"reply": ("pre", "early", "mean"), "user": ("mean",)}
ALARM_FPR = 0.05
MIN_POSITIVES = 5


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def recall_at_fpr(y, s, fpr=ALARM_FPR):
    y, s = np.asarray(y), np.asarray(s)
    if not 0 < y.sum() < len(y):
        return float("nan")
    return float((s[y == 1] > np.quantile(s[y == 0], 1 - fpr)).mean())


def bootstrap(y, s, groups, fn=auroc, n=1000, seed=0):
    """95% interval, resampling whole groups so that items of one template stay together."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    members = {g: np.flatnonzero(groups == g) for g in set(groups.tolist())}
    keys = sorted(members); rng = np.random.default_rng(seed); values = []
    for _ in range(n):
        idx = np.concatenate([members[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        v = fn(y[idx], s[idx])
        if not np.isnan(v):
            values.append(v)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [float("nan")] * 2


def text_scores(train_texts, train_y, test_texts):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(vec.fit_transform(train_texts), train_y)
    return clf.predict_proba(vec.transform(test_texts))[:, 1]


def _held_out(group_key, fraction=0.2):
    return int(hashlib.sha256(group_key.encode()).hexdigest(), 16) % 1000 < fraction * 1000


def _gkey(r):
    return f"{r['source']}:{r['group']}"


def evaluate(records, feats, *, site, pool, layer, split="val", detectors=DEFAULT_DETECTORS, n_boot=1000,
             references=True):
    """One site, one pooling, one layer -> metrics per detector and concept."""
    rs = [r for r in records if r["site"] == site and r["id"] in feats]
    X = lambda items: feats.get([r["id"] for r in items], pool, layer)
    dev_normal = [r for r in rs if r["split"] == "dev" and is_normal(r)]
    fit_set = [r for r in dev_normal if not _held_out(_gkey(r))]
    calib = [r for r in dev_normal if _held_out(_gkey(r))]
    eval_set = [r for r in rs if r["split"] == split]
    eval_normal = [r for r in eval_set if is_normal(r)]
    out = dict(site=site, pool=pool, layer=layer, split=split, n_fit=len(fit_set), n_calib=len(calib),
               n_eval_normal=len(eval_normal), detectors={}, references={})
    if len(fit_set) < 20 or len(calib) < 5 or len(eval_normal) < 5:
        out["skipped"] = "too few normal records"
        return out

    def concept_sets(concept):
        pos = [r for r in eval_set if is_positive(r, concept)]
        sources = {r["source"] for r in pos}
        neg = [r for r in eval_normal if r["source"] in sources]
        return pos, neg

    X_fit, X_calib, X_eval_normal = X(fit_set), X(calib), X(eval_normal)
    for name in detectors:
        det = ONE_CLASS[name]().fit(X_fit, groups=[_gkey(r) for r in fit_set])
        s_calib, s_new = det.score(X_calib), det.score(X_eval_normal)
        threshold = float(np.quantile(s_calib, 1 - ALARM_FPR))
        res = dict(threshold=threshold, fpr_new_normal=float((s_new > threshold).mean()),
                   shift_auroc=auroc([0] * len(s_calib) + [1] * len(s_new), np.concatenate([s_calib, s_new])),
                   concepts={})
        for concept in CONCEPTS:
            pos, neg = concept_sets(concept)
            if len(pos) < MIN_POSITIVES or len(neg) < MIN_POSITIVES:
                continue
            items = pos + neg; y = [1] * len(pos) + [0] * len(neg); s = det.score(X(items))
            res["concepts"][concept] = dict(
                n_pos=len(pos), n_neg=len(neg), auroc=auroc(y, s),
                ci=bootstrap(y, s, [_gkey(r) for r in items], n=n_boot) if n_boot else None,
                recall_at_fpr5=recall_at_fpr(y, s), recall_at_threshold=float((s[:len(pos)] > threshold).mean()))
        res["mean_auroc"] = float(np.mean([c["auroc"] for c in res["concepts"].values()])) if res["concepts"] else float("nan")
        out["detectors"][name] = res

    if references:
        dev = [r for r in rs if r["split"] == "dev"]
        for concept in CONCEPTS:
            pos, neg = concept_sets(concept)
            tr_pos = [r for r in dev if is_positive(r, concept)]
            tr_neg = [r for r in dev_normal if r["source"] in {p["source"] for p in tr_pos}]
            if min(len(pos), len(neg), len(tr_pos), len(tr_neg)) < MIN_POSITIVES:
                continue
            items = pos + neg; y = [1] * len(pos) + [0] * len(neg)
            ty = [1] * len(tr_pos) + [0] * len(tr_neg)
            probe = LinearProbe().fit(X(tr_pos + tr_neg), ty)
            out["references"][concept] = dict(
                probe_auroc=auroc(y, probe.score(X(items))),
                text_auroc=auroc(y, text_scores([r["target"] for r in tr_pos + tr_neg], ty, [r["target"] for r in items])))
    return out


FEWSHOT_KS = (1, 2, 5, 10, 20, None)          # None = every positive example in the dev split


def fewshot_settings(feats):
    """Fixed in advance, not selected on labels: the reply mean at mid-depth, the pre-reply state at the deepest
    extracted layer, and the user message mean at mid-depth."""
    mid, deep = feats.layers[len(feats.layers) // 2], feats.layers[-1]
    return [("reply", "mean", mid), ("reply", "pre", deep), ("user", "mean", mid)]


def fewshot(records, feats, *, split="val", ks=FEWSHOT_KS, draws=20, seed=0, max_normal=None):
    """How many labelled examples does a concept need? For each concept and k: draw k positive dev examples, build
    the detector, score the evaluated split; repeat `draws` times.
      few_shot  whitened mean difference; geometry from ALL normal dev records of the site
      probe     logistic regression on the same k positives against the same-source normal dev records
      text      character n-gram classifier on the same examples
    Positives and negatives of a concept come from the same source, as in evaluate().
    max_normal: fit the geometry on a random subset of that many normal records (effect of the normal corpus size)."""
    from .detectors import FewShotDirection
    out = dict(split=split, model=feats.meta.get("model"), draws=draws, runs=[])
    for site, pool, layer in fewshot_settings(feats):
        rs = [r for r in records if r["site"] == site and r["id"] in feats]
        X = lambda items: feats.get([r["id"] for r in items], pool, layer)
        normal = [r for r in rs if r["split"] == "dev" and is_normal(r)]
        if len(normal) < 20:
            continue
        geometry = normal
        if max_normal and max_normal < len(normal):
            pick = np.random.default_rng(seed).choice(len(normal), max_normal, replace=False)
            geometry = [normal[i] for i in pick]
        base = FewShotDirection().fit_normal(X(geometry))
        run = dict(site=site, pool=pool, layer=layer, n_normal=len(geometry), concepts={})
        for source in sorted({r["source"] for r in rs}):
            for concept in CONCEPTS:
                tr_pos = [r for r in rs if r["split"] == "dev" and r["source"] == source and is_positive(r, concept)]
                tr_neg = [r for r in normal if r["source"] == source]
                pos = [r for r in rs if r["split"] == split and r["source"] == source and is_positive(r, concept)]
                neg = [r for r in rs if r["split"] == split and r["source"] == source and is_normal(r)]
                if min(len(tr_pos), len(tr_neg), len(pos), len(neg)) < MIN_POSITIVES:
                    continue
                y = [1] * len(pos) + [0] * len(neg)
                X_pos, X_neg, X_eval = X(tr_pos), X(tr_neg), X(pos + neg)
                neg_text, eval_text = [r["target"] for r in tr_neg], [r["target"] for r in pos + neg]
                rng = np.random.default_rng(seed); curve = {}
                for k in ks:
                    n = len(tr_pos) if k is None else k
                    if n > len(tr_pos):
                        continue
                    scores = {"few_shot": [], "probe": [], "text": []}
                    for _ in range(1 if n == len(tr_pos) else draws):
                        idx = rng.choice(len(tr_pos), n, replace=False)
                        scores["few_shot"].append(auroc(y, base.fit_concept(X_pos[idx], X_neg).score(X_eval)))
                        labels = [1] * n + [0] * len(tr_neg)
                        probe = LinearProbe().fit(np.concatenate([X_pos[idx], X_neg]), labels)
                        scores["probe"].append(auroc(y, probe.score(X_eval)))
                        scores["text"].append(auroc(y, text_scores([tr_pos[i]["target"] for i in idx] + neg_text,
                                                                    labels, eval_text)))
                    curve["all" if k is None else str(k)] = dict(
                        n=n, **{name: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for name, v in scores.items()})
                run["concepts"][f"{source}/{concept}"] = dict(n_pos=len(pos), n_neg=len(neg), n_dev_pos=len(tr_pos),
                                                               curve=curve)
        out["runs"].append(run)
    return out


def token_basis(records, feats, *, depth_index=None, components=128):
    """Principal components of the normal dev records (reply and user sites together, mean pooling) at one layer.
    extract.py saves every token in these coordinates. Only the subspace is fixed here; whitening is redone later."""
    from .detectors import FewShotDirection
    layer = feats.layers[len(feats.layers) // 2 if depth_index is None else depth_index]
    normal = [r for r in records if r["split"] == "dev" and is_normal(r) and r["id"] in feats]
    base = FewShotDirection(components=components).fit_normal(feats.get([r["id"] for r in normal], "mean", layer))
    return dict(layer=layer, mean=base.scaler.mean, scale=base.scaler.scale, components=base.pca.components_)


TOKEN_POOLS = {"mean": lambda s: s.mean(), "max": lambda s: s.max(), "top5": lambda s: np.sort(s)[-5:].mean()}


def tokens(records, feats, *, split="val", ks=(5, None), draws=20, seed=0):
    """Does reading token by token beat the mean? The direction is built as in fewshot() from the per-record mean
    of the saved token coordinates; each record is then scored by the mean, the maximum, or the mean of its five
    highest token scores."""
    from .detectors import FewShotDirection
    out = dict(split=split, model=feats.meta.get("model"), layer=feats.meta.get("token_layer"), draws=draws, concepts={})
    for site in ("reply", "user"):
        rs = [r for r in records if r["site"] == site and r["id"] in feats]
        T = dict(zip([r["id"] for r in rs], feats.tokens([r["id"] for r in rs])))
        M = lambda items: np.stack([T[r["id"]].mean(0) for r in items])
        normal = [r for r in rs if r["split"] == "dev" and is_normal(r)]
        if len(normal) < 20:
            continue
        base = FewShotDirection(components=10 ** 6).fit_normal(M(normal))
        for source in sorted({r["source"] for r in rs}):
            for concept in CONCEPTS:
                tr_pos = [r for r in rs if r["split"] == "dev" and r["source"] == source and is_positive(r, concept)]
                tr_neg = [r for r in normal if r["source"] == source]
                pos = [r for r in rs if r["split"] == split and r["source"] == source and is_positive(r, concept)]
                neg = [r for r in rs if r["split"] == split and r["source"] == source and is_normal(r)]
                if min(len(tr_pos), len(tr_neg), len(pos), len(neg)) < MIN_POSITIVES:
                    continue
                y = [1] * len(pos) + [0] * len(neg)
                X_pos, X_neg = M(tr_pos), M(tr_neg)
                rng = np.random.default_rng(seed); res = {}
                for k in ks:
                    n = len(tr_pos) if k is None else k
                    if n > len(tr_pos):
                        continue
                    scores = {name: [] for name in TOKEN_POOLS}
                    for _ in range(1 if n == len(tr_pos) else draws):
                        base.fit_concept(X_pos[rng.choice(len(tr_pos), n, replace=False)], X_neg)
                        per_token = [base.score(T[r["id"]]) for r in pos + neg]
                        for name, pool in TOKEN_POOLS.items():
                            scores[name].append(auroc(y, [pool(t) for t in per_token]))
                    res["all" if k is None else str(k)] = {name: float(np.mean(v)) for name, v in scores.items()}
                out["concepts"][f"{site}/{source}/{concept}"] = dict(n_pos=len(pos), n_neg=len(neg), k=res)
    return out


def contrast(records, feats, *, split="val", ks=(5, 20, None), draws=20, seed=0):
    """Can a concept be defined by generated examples alone? Fit on the "syn" contrast pairs (the model's own reply
    against its reply written to show the concept), test on the REAL labelled records of `split`.
      few_shot  whitened difference between the two halves of the pairs; geometry from the normal dev records
      probe     logistic regression on the pairs
      text      character n-gram classifier on the pairs
    The pre-reply state is the same for both halves of a pair, so only pools read on the reply are used."""
    from .detectors import FewShotDirection
    mid = feats.layers[len(feats.layers) // 2]
    rs = [r for r in records if r["site"] == "reply" and r["id"] in feats]
    out = dict(split=split, model=feats.meta.get("model"), draws=draws, runs=[])
    for pool in ("mean", "early"):
        X = lambda items: feats.get([r["id"] for r in items], pool, mid)
        normal = [r for r in rs if r["split"] == "dev" and is_normal(r)]
        base = FewShotDirection().fit_normal(X(normal))
        run = dict(pool=pool, layer=mid, n_normal=len(normal), concepts={})
        for concept in CONCEPTS:
            syn = [r for r in rs if r["split"] == "syn" and r["meta"]["concept"] == concept]
            s_pos = [r for r in syn if is_positive(r, concept)]
            s_neg = [r for r in syn if not is_positive(r, concept)]
            if min(len(s_pos), len(s_neg)) < MIN_POSITIVES:
                continue
            Xp, Xn = X(s_pos), X(s_neg)
            for source in sorted({r["source"] for r in rs}):
                pos = [r for r in rs if r["split"] == split and r["source"] == source and is_positive(r, concept)]
                neg = [r for r in rs if r["split"] == split and r["source"] == source and is_normal(r)]
                if min(len(pos), len(neg)) < MIN_POSITIVES:
                    continue
                y = [1] * len(pos) + [0] * len(neg); X_eval = X(pos + neg)
                eval_text = [r["target"] for r in pos + neg]
                rng = np.random.default_rng(seed); curve = {}
                for k in ks:
                    n = min(len(s_pos), len(s_neg)) if k is None else k
                    if n > min(len(s_pos), len(s_neg)):
                        continue
                    scores = {"few_shot": [], "probe": [], "text": []}
                    for _ in range(1 if k is None else draws):
                        i, j = rng.choice(len(s_pos), n, replace=False), rng.choice(len(s_neg), n, replace=False)
                        labels = [1] * n + [0] * n
                        scores["few_shot"].append(auroc(y, base.fit_concept(Xp[i], Xn[j]).score(X_eval)))
                        scores["probe"].append(auroc(y, LinearProbe().fit(np.concatenate([Xp[i], Xn[j]]), labels).score(X_eval)))
                        if n >= 5:
                            scores["text"].append(auroc(y, text_scores([s_pos[a]["target"] for a in i] + [s_neg[b]["target"] for b in j],
                                                                        labels, eval_text)))
                    curve["all" if k is None else str(k)] = dict(n=n, **{m: float(np.mean(v)) for m, v in scores.items() if v})
                run["concepts"][f"{source}/{concept}"] = dict(n_pos=len(pos), n_neg=len(neg), n_pairs=len(s_pos), curve=curve)
        out["runs"].append(run)
    return out


def sealed(records, feats, *, n_boot=1000, draws=20, seed=0):
    """The preregistered test (PREREG_V2.md). Everything is fitted on dev / generated data; the test split is only
    scored. Hostility is evaluated on the new sealed set "hx" and on the old test split of "c1", overall and WITHIN
    the insulting-history level, where the conversation history is the same kind for positives and negatives."""
    from .detectors import FewShotDirection
    layers = feats.layers; mid, deep = layers[len(layers) // 2], layers[-1]
    rs = [r for r in records if r["site"] == "reply" and r["id"] in feats]
    X = lambda items, pool, layer: feats.get([r["id"] for r in items], pool, layer)
    normal = [r for r in rs if r["split"] == "dev" and is_normal(r)]
    base = {(pool, layer): FewShotDirection().fit_normal(X(normal, pool, layer))
            for pool, layer in (("early", mid), ("mean", mid), ("pre", deep))}
    syn = lambda concept, label: [r for r in rs if r["split"] == "syn" and r["meta"]["concept"] == concept
                                  and is_positive(r, concept) == label]
    real = lambda source, concept, label: [r for r in rs if r["split"] == "dev" and r["source"] == source and
                                           (is_positive(r, concept) if label else is_normal(r))]

    def direction(pos, neg, pool, layer):
        b = base[(pool, layer)]
        w = b._white(X(pos, pool, layer)).mean(0) - b._white(X(neg, pool, layer)).mean(0)
        return lambda items: b._white(X(items, pool, layer)) @ w

    def text(pos, neg):
        return lambda items: text_scores([r["target"] for r in pos + neg], [1] * len(pos) + [0] * len(neg),
                                         [r["target"] for r in items])

    def few(pos, neg, pool, layer, k):
        rng = np.random.default_rng(seed)
        fns = [direction([pos[i] for i in rng.choice(len(pos), k, replace=False)], neg, pool, layer) for _ in range(draws)]
        return lambda items: np.mean([_rank(f(items)) for f in fns], axis=0)       # mean rank over the draws

    def report(items, concept, methods, paired=()):
        y = np.array([int(is_positive(r, concept)) for r in items]); groups = [_gkey(r) for r in items]
        out = dict(n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), methods={})
        if min(out["n_pos"], out["n_neg"]) < MIN_POSITIVES:
            return out
        scores = {name: np.asarray(fn(items), dtype=float) for name, fn in methods.items()}
        for name, sc in scores.items():
            out["methods"][name] = dict(auroc=auroc(y, sc), ci=bootstrap(y, sc, groups, n=n_boot),
                                        recall_at_fpr5=recall_at_fpr(y, sc))
        for a, b in paired:
            out[f"{a}_minus_{b}"] = _paired(y, scores[a], scores[b], groups, n_boot)
        return out

    result = dict(model=feats.meta.get("model"), n_normal=len(normal), hostile={}, other={})
    sp, sn = syn("hostile", True), syn("hostile", False)
    rp, rn = real("c1", "hostile", True), real("c1", "hostile", False)
    probe = LinearProbe().fit(X(rp + rn, "mean", mid), [1] * len(rp) + [0] * len(rn))
    methods = {"syn_early": direction(sp, sn, "early", mid), "syn_mean": direction(sp, sn, "mean", mid),
               "text_syn": text(sp, sn), "real5_mean": few(rp, rn, "mean", mid, 5),
               "real_all_mean": direction(rp, rn, "mean", mid), "probe_mean": lambda items: probe.score(X(items, "mean", mid)),
               "pre_all": direction(rp, rn, "pre", deep), "text_real": text(rp, rn)}
    for source in ("hx", "c1"):
        test = [r for r in rs if r["split"] == "test" and r["source"] == source]
        within = [r for r in test if r["meta"]["level"] == "insulting"]
        paired = (("syn_early", "text_syn"),)
        result["hostile"][source] = dict(overall=report(test, "hostile", methods, paired),
                                         within_insulting=report(within, "hostile", methods, paired))
    # Secondary, no thresholds: the old test splits of the emotion and sycophancy sets.
    for concept in ("anger", "joy", "sadness"):
        test = [r for r in rs if r["split"] == "test" and r["source"] == "emo" and (is_positive(r, concept) or is_normal(r))]
        ep, en = syn(concept, True), syn(concept, False)
        if min(len(ep), len(en), len(test)) < MIN_POSITIVES:
            continue
        result["other"][f"emo/{concept}"] = report(test, concept, {"syn_mean": direction(ep, en, "mean", mid), "text_syn": text(ep, en)})
    test = [r for r in rs if r["split"] == "test" and r["source"] == "syco"]
    yp, yn = real("syco", "syco", True), real("syco", "syco", False)
    ep, en = syn("syco", True), syn("syco", False)
    if min(len(yp), len(yn), len(ep), len(en), len(test)) >= MIN_POSITIVES:
        result["other"]["syco/syco"] = report(test, "syco", {"syn_mean": direction(ep, en, "mean", mid),
                                                              "real_all_mean": direction(yp, yn, "mean", mid),
                                                              "pre_all": direction(yp, yn, "pre", deep), "text_real": text(yp, yn)})
    return result


def _rank(s):
    return np.argsort(np.argsort(s)) / max(1, len(s) - 1)


def _paired(y, a, b, groups, n=1000, seed=0):
    """AUROC(a) - AUROC(b) with a 95% interval, resampling whole groups."""
    groups = np.asarray(groups); members = {g: np.flatnonzero(groups == g) for g in set(groups.tolist())}
    keys = sorted(members); rng = np.random.default_rng(seed); values = []
    for _ in range(n):
        idx = np.concatenate([members[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        d = auroc(y[idx], a[idx]) - auroc(y[idx], b[idx])
        if not np.isnan(d):
            values.append(d)
    return dict(diff=auroc(y, a) - auroc(y, b), ci=[float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else None)


def fewshot_table(result, method="few_shot"):
    ks = ["1", "2", "5", "10", "20", "all"]
    lines = [f"| site | pool | layer | concept | eval pos | " + " | ".join("k=" + k for k in ks) + " |",
             "|---|---|---|---|---|" + "---|" * len(ks)]
    for run in result["runs"]:
        for name, c in run["concepts"].items():
            cells = [f"{c['curve'][k][method]['mean']:.3f}" if k in c["curve"] else "" for k in ks]
            lines.append(f"| {run['site']} | {run['pool']} | {run['layer']} | {name} | {c['n_pos']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sweep(records, feats, *, split="val", detectors=DEFAULT_DETECTORS, n_boot=0):
    runs = []
    for site, pools in SITE_POOLS.items():
        for pool in pools:
            for layer in feats.layers:
                runs.append(evaluate(records, feats, site=site, pool=pool, layer=layer, split=split,
                                     detectors=detectors, n_boot=n_boot, references=False))
    return dict(split=split, model=feats.meta.get("model"), runs=runs)


def freeze(sweep_result):
    """Per site and detector: the pooling and layer with the highest mean AUROC over concepts on the validation
    split. Ties go to the earlier pooling and shallower layer (sweep order)."""
    if sweep_result["split"] != "val":
        raise ValueError("freeze from the validation sweep only")
    chosen = {}
    for run in sweep_result["runs"]:
        for name, res in run["detectors"].items():
            key = f"{run['site']}/{name}"
            if not np.isnan(res["mean_auroc"]) and (key not in chosen or res["mean_auroc"] > chosen[key]["val_mean_auroc"] + 1e-9):
                chosen[key] = dict(site=run["site"], detector=name, pool=run["pool"], layer=run["layer"],
                                   val_mean_auroc=res["mean_auroc"])
    return dict(model=sweep_result.get("model"), rule="max mean val AUROC over concepts", chosen=chosen)


def run_frozen(records, feats, frozen, *, split="test", n_boot=1000):
    return dict(split=split, model=feats.meta.get("model"), runs=[
        evaluate(records, feats, site=c["site"], pool=c["pool"], layer=c["layer"], split=split,
                 detectors=(c["detector"],), n_boot=n_boot) for c in frozen["chosen"].values()])


def table(result):
    """Markdown: one row per detector setting, one column per concept (AUROC), plus the topic-shift control."""
    lines = ["| site | detector | pool | layer | " + " | ".join(CONCEPTS) + " | FPR on new normal | shift AUROC |",
             "|---|---|---|---|" + "---|" * (len(CONCEPTS) + 2)]
    shown = set()                                   # the supervised references depend on the setting, not the detector
    for run in result["runs"]:
        for name, res in run["detectors"].items():
            cells = [f"{res['concepts'][c]['auroc']:.3f}" if c in res["concepts"] else "" for c in CONCEPTS]
            lines.append(f"| {run['site']} | {name} | {run['pool']} | {run['layer']} | " + " | ".join(cells)
                         + f" | {res['fpr_new_normal']:.3f} | {res['shift_auroc']:.3f} |")
        key = (run["site"], run["pool"], run["layer"])
        if run["references"] and key not in shown:
            shown.add(key)
            for kind in ("probe_auroc", "text_auroc"):
                cells = [f"{run['references'][c][kind]:.3f}" if c in run["references"] else "" for c in CONCEPTS]
                lines.append(f"| {run['site']} | *{kind.split('_')[0]} (supervised)* | {run['pool']} | {run['layer']} | "
                             + " | ".join(cells) + " | | |")
    return "\n".join(lines)


def _write(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)


def main(argv=None):
    from .extract import Features
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fewshot")
    p.add_argument("--records", required=True); p.add_argument("--feats", required=True)
    p.add_argument("--out", required=True); p.add_argument("--draws", type=int, default=20)
    p.add_argument("--max-normal", type=int)
    for name in ("basis", "tokens", "contrast", "sealed"):
        p = sub.add_parser(name)
        p.add_argument("--records", required=True); p.add_argument("--feats", required=True, nargs="+")
        p.add_argument("--out", required=True)
    for name in ("sweep", "test"):
        p = sub.add_parser(name)
        p.add_argument("--records", required=True); p.add_argument("--feats", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--detectors", nargs="*", default=list(DEFAULT_DETECTORS), choices=sorted(ONE_CLASS))
        if name == "test":
            p.add_argument("--freeze", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--sweep", required=True); p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "freeze":
        with open(args.sweep) as f:
            frozen = freeze(json.load(f))
        if os.path.exists(args.out):
            raise SystemExit("freeze file already exists; a freeze is written once")
        _write(args.out, frozen); print(json.dumps(frozen, indent=1)); return
    if args.command == "sealed" and os.path.exists(args.out):
        raise SystemExit("sealed test results already exist; the test split is opened once")
    records = load(args.records)
    feats = Features(*args.feats) if isinstance(args.feats, list) else Features(args.feats)
    if args.command == "fewshot":
        result = fewshot(records, feats, draws=args.draws, max_normal=args.max_normal)
        _write(args.out, result)
        for method in ("few_shot", "probe", "text"):
            print(f"\n{method}\n" + fewshot_table(result, method))
        return
    if args.command == "basis":
        np.savez(args.out, **token_basis(records, feats)); return
    if args.command in ("tokens", "contrast", "sealed"):
        result = {"tokens": tokens, "contrast": contrast, "sealed": sealed}[args.command](records, feats)
        _write(args.out, result); print(json.dumps(result, ensure_ascii=False, indent=1)); return
    if args.command == "sweep":
        result = sweep(records, feats, detectors=tuple(args.detectors))
    else:
        if os.path.exists(args.out):
            raise SystemExit("test results already exist; the test split is opened once")
        with open(args.freeze) as f:
            result = run_frozen(records, feats, json.load(f))
    _write(args.out, result); print(table(result))


if __name__ == "__main__":
    main()
