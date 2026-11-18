import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nous_anomaly.detectors import (AutoEncoder, ConceptSubspace, ContentFreeMahalanobis, IsoForest, KNN, LinearProbe,
                                    Mahalanobis)
from synth import World


def _auc(det, normal, odd):
    s = np.concatenate([det.score(normal), det.score(odd)])
    return roc_auc_score([0] * len(normal) + [1] * len(odd), s)


@pytest.fixture(scope="module")
def world():
    w = World(0)
    X, groups = w.normal()
    seen_topic = w.topic_mean()
    fresh = w.sample(300, seen_topic)
    return dict(w=w, X=X, groups=groups, fresh=fresh,
                risky=w.sample(300, seen_topic, shift=8.0 * w.risk),
                far_topic=w.sample(300, w.topic_mean(scale=15.0)))


@pytest.mark.parametrize("cls", [Mahalanobis, ContentFreeMahalanobis, KNN, IsoForest])
def test_one_class_detectors_flag_a_shift_they_never_saw(world, cls):
    det = cls().fit(world["X"], groups=world["groups"])
    assert _auc(det, world["fresh"], world["risky"]) > (0.95 if isinstance(det, Mahalanobis) else 0.85)


def test_plain_mahalanobis_fires_on_a_far_topic_and_content_free_does_not(world):
    plain = Mahalanobis().fit(world["X"])
    free = ContentFreeMahalanobis().fit(world["X"], groups=world["groups"])
    # The far topic sits 2.5x further out than any training topic. Content removal is estimated from data, so a
    # little of the topic leaks through: the alarm drops a lot but not to chance.
    assert _auc(plain, world["fresh"], world["far_topic"]) > 0.95
    assert _auc(free, world["fresh"], world["far_topic"]) < 0.75
    assert _auc(free, world["fresh"], world["risky"]) > 0.95


def test_content_free_without_groups_drops_top_components(world):
    free = ContentFreeMahalanobis(drop=8).fit(world["X"])
    assert free.content.shape == (8, world["X"].shape[1])
    assert _auc(free, world["fresh"], world["far_topic"]) < 0.75


def test_scores_are_deterministic_and_finite(world):
    a = ContentFreeMahalanobis().fit(world["X"], groups=world["groups"]).score(world["risky"])
    b = ContentFreeMahalanobis().fit(world["X"], groups=world["groups"]).score(world["risky"])
    assert np.isfinite(a).all() and np.allclose(a, b)


def test_few_samples_high_dimension_does_not_crash():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 500))
    for cls in (Mahalanobis, ContentFreeMahalanobis, KNN, IsoForest):
        assert np.isfinite(cls().fit(X).score(rng.normal(size=(5, 500)))).all()


def test_concept_subspace_separates_known_from_unknown_deviation(world):
    w = world["w"]
    det = ConceptSubspace([w.risk]).fit(world["X"], groups=world["groups"])
    topic = w.topic_mean()
    base = w.sample(300, topic)
    known = w.sample(300, topic, shift=8.0 * w.risk)
    unknown = w.sample(300, topic, shift=8.0 * w.other)
    auc = lambda fn, odd: roc_auc_score([0] * 300 + [1] * 300, np.concatenate([fn(base), fn(odd)]))
    assert auc(det.score_inside, known) > 0.95 and auc(det.score_outside, known) < 0.65
    assert auc(det.score_outside, unknown) > 0.9 and auc(det.score_inside, unknown) < 0.65


def test_linear_probe_binary_and_probespec(world):
    X = np.concatenate([world["fresh"], world["risky"]]); y = [0] * 300 + [1] * 300
    probe = LinearProbe().fit(X[::2], y[::2])
    assert roc_auc_score(y[1::2], probe.score(X[1::2])) > 0.95
    spec = probe.to_probespec("C1", "c1_mean", 18, "mean_span", note="x")
    assert set(spec) == {"concept", "name", "layer", "pool", "mean", "scale", "coef", "intercept", "threshold", "meta"}
    # Same arithmetic as monitor/core/probe.py: sigmoid(((h - mean) / scale) @ coef + intercept)
    z = ((X[1] - np.array(spec["mean"])) / np.array(spec["scale"])) @ np.array(spec["coef"]) + spec["intercept"]
    assert np.isclose(1 / (1 + np.exp(-z)), probe.score(X[1:2])[0])
    cos = probe.direction()[0] @ world["w"].risk / np.linalg.norm(probe.direction()[0])
    assert cos > 0.5


def test_linear_probe_multiclass_scores_departure_from_neutral(world):
    w = world["w"]; topic = w.topic_mean()
    parts = {"neutral": w.sample(200, topic), "anger": w.sample(200, topic, 5.0 * w.risk),
             "joy": w.sample(200, topic, 5.0 * w.other)}
    X = np.concatenate(list(parts.values())); y = sum(([k] * 200 for k in parts), [])
    probe = LinearProbe().fit(X[::2], y[::2])
    assert (probe.clf.predict(probe.scaler(X[1::2])) == np.array(y[1::2])).mean() > 0.9
    s = probe.score(X[1::2]); is_neutral = np.array(y[1::2]) == "neutral"
    assert s[~is_neutral].mean() > 0.8 and s[is_neutral].mean() < 0.2


def test_autoencoder_if_torch_is_installed(world):
    pytest.importorskip("torch")
    det = AutoEncoder(epochs=15, bottleneck=16, hidden=64).fit(world["X"])
    assert _auc(det, world["fresh"], world["risky"]) > 0.8
