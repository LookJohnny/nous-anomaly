"""Synthetic activations with the structure E02 found: content dominates the variance, the risky signal is small."""
import numpy as np

D, CONTENT_DIMS = 96, 8


class World:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        basis, _ = np.linalg.qr(self.rng.normal(size=(D, D)))
        self.content = basis[:, :CONTENT_DIMS]              # topics differ along these directions
        self.risk = basis[:, CONTENT_DIMS]                  # a low-variance direction, orthogonal to content
        self.other = basis[:, CONTENT_DIMS + 1]

    def topic_mean(self, scale=6.0):
        return self.content @ self.rng.normal(scale=scale, size=CONTENT_DIMS)

    def sample(self, n, topic_mean, shift=None):
        x = topic_mean + self.rng.normal(scale=1.0, size=(n, D))
        return x + shift if shift is not None else x

    def normal(self, n_topics=30, per_topic=40):
        xs, groups = [], []
        for t in range(n_topics):
            xs.append(self.sample(per_topic, self.topic_mean())); groups += [f"t{t}"] * per_topic
        return np.concatenate(xs), groups
