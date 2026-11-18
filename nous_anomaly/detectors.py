"""Detectors over activation vectors. Higher score = more anomalous.

One-class detectors see NORMAL vectors only:   det.fit(X_normal[, groups=topic_ids]);  det.score(X)
The supervised probe is the reference upper bound: probe.fit(X, y);  probe.score(X)

Why "content-free": in these models content explains about 90% of the activation variance and emotion a few percent
(E02). A detector on the raw space mostly fires on a new topic. ContentFreeMahalanobis removes the directions along
which normal data varies by topic before measuring distance.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class _Scaler:
    def fit(self, X):
        self.mean = X.mean(0)
        self.scale = X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.scale


def _zfit(values):
    return float(np.mean(values)), float(np.std(values) + 1e-9)


class Mahalanobis:
    """Distance from the normal cloud: whitened distance inside the top principal components, plus how much of the
    vector lies outside them. Each part is z-scored on the normal data and the two are added."""
    name = "mahalanobis"

    def __init__(self, components=256, shrink=0.1, residual_weight=1.0):
        self.components, self.shrink, self.residual_weight = components, shrink, residual_weight

    def _project(self, Z):
        return Z

    def fit(self, X, groups=None):
        self.scaler = _Scaler().fit(np.asarray(X, dtype=np.float64))
        Z = self._fit_projection(self.scaler(X), groups)
        k = int(min(self.components, Z.shape[0] - 1, Z.shape[1]))
        self.pca = PCA(n_components=k, svd_solver="auto", random_state=0).fit(Z)
        var = self.pca.explained_variance_
        self.var = var + self.shrink * var.mean()          # shrinkage: tiny eigenvalues are mostly noise
        inside, outside = self._parts(Z)
        self.z_in, self.z_out = _zfit(inside), _zfit(outside)
        # When the components span the whole space nothing lies outside them; the residual is rounding noise.
        self.has_outside = bool(np.mean(outside) > 1e-6 * np.linalg.norm(Z, axis=1).mean())
        return self

    def _fit_projection(self, Z, groups):
        return Z

    def _parts(self, Z):
        T = self.pca.transform(Z)
        inside = np.sqrt((T ** 2 / self.var).sum(1))
        outside = np.linalg.norm(Z - self.pca.inverse_transform(T), axis=1)
        return inside, outside

    def score(self, X):
        inside, outside = self._parts(self._project(self.scaler(X)))
        score = (inside - self.z_in[0]) / self.z_in[1]
        if self.has_outside:
            score = score + self.residual_weight * (outside - self.z_out[0]) / self.z_out[1]
        return score


class ContentFreeMahalanobis(Mahalanobis):
    """Mahalanobis after removing content directions.

    With `groups` (a topic or scenario id per normal vector): the directions along which the group means differ are
    removed, as many as needed to explain `between` of the between-group variance (at most `drop`).
    Without groups: the top `drop` principal components of the normal data are removed."""
    name = "content_free"

    def __init__(self, drop=32, between=0.95, **kw):
        super().__init__(**kw)
        self.drop, self.between = drop, between

    def _fit_projection(self, Z, groups):
        if groups is not None and len(set(groups)) > 2:
            groups = np.asarray(groups)
            means = np.stack([Z[groups == g].mean(0) for g in sorted(set(groups.tolist()))])
            _, s, Vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
            share = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
            k = int(min(self.drop, np.searchsorted(share, self.between) + 1, len(s)))
        else:
            k = int(min(self.drop, Z.shape[0] - 1, Z.shape[1] - 1))
            Vt = PCA(n_components=k, random_state=0).fit(Z).components_
        self.content = Vt[:k]
        return self._project(Z)

    def _project(self, Z):
        return Z - (Z @ self.content.T) @ self.content


class KNN:
    """Mean distance to the k nearest normal vectors, in a reduced space."""
    name = "knn"

    def __init__(self, k=10, components=64):
        self.k, self.components = k, components

    def fit(self, X, groups=None):
        self.scaler = _Scaler().fit(np.asarray(X, dtype=np.float64))
        Z = self.scaler(X)
        self.pca = PCA(n_components=int(min(self.components, Z.shape[0] - 1, Z.shape[1])), random_state=0).fit(Z)
        T = self.pca.transform(Z)
        self.k_ = int(min(self.k, len(T) - 1))
        self.nn = NearestNeighbors(n_neighbors=self.k_ + 1).fit(T)
        return self

    def score(self, X):
        dist, _ = self.nn.kneighbors(self.pca.transform(self.scaler(X)), n_neighbors=self.k_)
        return dist.mean(1)


class IsoForest:
    name = "isolation_forest"

    def __init__(self, components=64, trees=300):
        self.components, self.trees = components, trees

    def fit(self, X, groups=None):
        self.scaler = _Scaler().fit(np.asarray(X, dtype=np.float64))
        Z = self.scaler(X)
        self.pca = PCA(n_components=int(min(self.components, Z.shape[0] - 1, Z.shape[1])), random_state=0).fit(Z)
        self.forest = IsolationForest(n_estimators=self.trees, random_state=0).fit(self.pca.transform(Z))
        return self

    def score(self, X):
        return -self.forest.score_samples(self.pca.transform(self.scaler(X)))


class AutoEncoder:
    """Reconstruction error of a small bottleneck network trained on normal vectors. Needs torch."""
    name = "autoencoder"

    def __init__(self, bottleneck=64, hidden=512, epochs=30, lr=1e-3, batch=256, device=None, seed=0):
        self.bottleneck, self.hidden, self.epochs, self.lr, self.batch = bottleneck, hidden, epochs, lr, batch
        self.device, self.seed = device, seed

    def fit(self, X, groups=None):
        import torch
        torch.manual_seed(self.seed)
        self.scaler = _Scaler().fit(np.asarray(X, dtype=np.float64))
        Z = torch.tensor(self.scaler(X), dtype=torch.float32)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = Z.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden), torch.nn.GELU(), torch.nn.Linear(self.hidden, self.bottleneck),
            torch.nn.GELU(), torch.nn.Linear(self.bottleneck, self.hidden), torch.nn.GELU(),
            torch.nn.Linear(self.hidden, d)).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(Z), generator=gen).split(self.batch):
                batch = Z[idx].to(device)
                loss = ((self.net(batch) - batch) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        self._device = device
        return self

    def score(self, X):
        import torch
        Z = torch.tensor(self.scaler(X), dtype=torch.float32)
        out = []
        with torch.no_grad():
            for chunk in Z.split(4096):
                chunk = chunk.to(self._device)
                out.append(((self.net(chunk) - chunk) ** 2).mean(1).cpu())
        return torch.cat(out).numpy().astype(np.float64)


class ConceptSubspace:
    """Splits the anomaly score in two: inside the span of known concept directions (e.g. probe weights) and outside
    it. A high outside score is a deviation none of the known probes explains."""
    name = "concept_subspace"

    def __init__(self, directions, outside=None):
        Q, _ = np.linalg.qr(np.asarray(directions, dtype=np.float64).T)
        self.Q = Q.T                                    # orthonormal rows
        self.outside = outside or ContentFreeMahalanobis()

    def fit(self, X, groups=None):
        X = np.asarray(X, dtype=np.float64)
        inside = X @ self.Q.T
        self.mean = inside.mean(0)
        cov = np.atleast_2d(np.cov(inside - self.mean, rowvar=False))
        self.precision = np.linalg.inv(cov + 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov)))
        self.outside.fit(X - inside @ self.Q, groups)
        return self

    def score_inside(self, X):
        diff = np.asarray(X, dtype=np.float64) @ self.Q.T - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.precision, diff))

    def score_outside(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.outside.score(X - (X @ self.Q.T) @ self.Q)

    def score(self, X):
        return self.score_outside(X)


class LinearProbe:
    """Supervised reference: standardise, then logistic regression (same recipe as the repo's ProbeSpec probes).
    Binary labels give P(positive); string labels give a multi-class probe and score() is 1 - P(normal_class)."""
    name = "linear_probe"

    def __init__(self, C=0.05, normal_class="neutral"):
        self.C, self.normal_class = C, normal_class

    def fit(self, X, y):
        self.scaler = _Scaler().fit(np.asarray(X, dtype=np.float64))
        self.clf = LogisticRegression(max_iter=2000, C=self.C).fit(self.scaler(X), np.asarray(y))
        self.classes = list(self.clf.classes_)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler(X))

    def score(self, X):
        proba = self.predict_proba(X)
        if len(self.classes) == 2:
            return proba[:, 1]
        return 1.0 - proba[:, self.classes.index(self.normal_class)]

    def direction(self):
        """Probe weights in the raw activation space, one row per class (one row for a binary probe)."""
        return self.clf.coef_ / self.scaler.scale

    def to_probespec(self, concept, name, layer, pool, **meta):
        """JSON-compatible with monitor/core/probe.py ProbeSpec (binary probes only)."""
        if len(self.classes) != 2:
            raise ValueError("ProbeSpec holds a binary probe")
        return dict(concept=concept, name=name, layer=int(layer), pool=pool, mean=self.scaler.mean.tolist(),
                    scale=self.scaler.scale.tolist(), coef=self.clf.coef_[0].tolist(),
                    intercept=float(self.clf.intercept_[0]), threshold=0.5, meta=meta)


class FewShotDirection:
    """A concept from a handful of examples. The normal data supplies the geometry (standardise, PCA, whiten with
    shrinkage); the direction is the mean of the k positive examples minus the normal mean in that whitened space.
    With the normal covariance this is Fisher's discriminant, so nothing has to be estimated from the k examples
    except their mean."""
    name = "few_shot"

    def __init__(self, components=128, shrink=0.1):
        self.components, self.shrink = components, shrink

    def fit_normal(self, X):
        self.scaler = _Scaler().fit(np.asarray(X, dtype=np.float64))
        Z = self.scaler(X)
        self.pca = PCA(n_components=int(min(self.components, Z.shape[0] - 1, Z.shape[1])), random_state=0).fit(Z)
        var = self.pca.explained_variance_
        self.sd = np.sqrt(var + self.shrink * var.mean())
        self.center = self._white(X).mean(0)
        return self

    def _white(self, X):
        return self.pca.transform(self.scaler(X)) / self.sd

    def fit_concept(self, X_positive, X_reference=None):
        """X_reference: normal examples from the same setting as the positives (default: all the normal data)."""
        center = self.center if X_reference is None else self._white(X_reference).mean(0)
        w = self._white(X_positive).mean(0) - center
        self.w, self.offset = w, float(center @ w)
        return self

    def fit(self, X_normal, X_positive, X_reference=None):
        return self.fit_normal(X_normal).fit_concept(X_positive, X_reference)

    def score(self, X):
        return self._white(X) @ self.w - self.offset


ONE_CLASS = {cls.name: cls for cls in (Mahalanobis, ContentFreeMahalanobis, KNN, IsoForest, AutoEncoder)}
