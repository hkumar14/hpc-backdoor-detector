"""One-class detectors.  Every detector is fitted on CLEAN log-counters only and returns an
anomaly score where larger = more anomalous.

All detectors work in a common *whitened log space*: counters are log-transformed (HPC counts
are multiplicative / ratio-like quantities, so a backdoor changes them by a factor) and then
decorrelated with a winsorised covariance so that no single counter dominates distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM


# --------------------------------------------------------------------------------------
# Whitening
# --------------------------------------------------------------------------------------
class Whitener:
    """Log -> centred -> decorrelated -> unit-variance transform (ZCA), robust to a few
    extreme clean samples (the clean file has a couple of 6-9 sigma cycle spikes)."""

    def __init__(self, winsor: float = 0.01):
        self.winsor = winsor

    def fit(self, Xlog: np.ndarray) -> "Whitener":
        lo, hi = np.quantile(Xlog, [self.winsor, 1 - self.winsor], axis=0)
        Xw = np.clip(Xlog, lo, hi)
        self.mean_ = Xw.mean(axis=0)
        cov = np.cov(Xw, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-18)
        self.W_ = vecs @ np.diag(vals ** -0.5) @ vecs.T
        return self

    def transform(self, Xlog: np.ndarray) -> np.ndarray:
        return (Xlog - self.mean_) @ self.W_


# --------------------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------------------
class BaseDetector:
    name = "base"

    def fit(self, Z: np.ndarray) -> "BaseDetector":  # Z is whitened log data
        raise NotImplementedError

    def score(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MahalanobisDetector(BaseDetector):
    """Distance under a single Gaussian (robust MCD fit).  Strong against shifts that break
    the correlation between counters; blind to multi-modality."""
    name = "mahalanobis"

    def __init__(self, support_fraction: float = 0.9, seed: int = 0):
        self.support_fraction, self.seed = support_fraction, seed

    def fit(self, Z):
        self.mcd_ = MinCovDet(support_fraction=self.support_fraction, random_state=self.seed).fit(Z)
        return self

    def score(self, Z):
        return np.sqrt(np.maximum(self.mcd_.mahalanobis(Z), 0.0))


class GMMDetector(BaseDetector):
    """Gaussian-mixture negative log-likelihood.  The clean traces are clearly multi-modal
    (two operating regimes), so a mixture models the *actual* clean density; the number of
    components is picked by BIC."""
    name = "gmm"

    def __init__(self, max_components: int = 4, seed: int = 0):
        self.max_components, self.seed = max_components, seed

    def fit(self, Z):
        best, best_bic = None, np.inf
        for k in range(1, self.max_components + 1):
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=3,
                                random_state=self.seed).fit(Z)
            b = g.bic(Z)
            if b < best_bic:
                best, best_bic = g, b
        self.gmm_ = best
        self.n_components_ = best.n_components
        return self

    def score(self, Z):
        return -self.gmm_.score_samples(Z)


class KNNDetector(BaseDetector):
    """Mean distance to the k nearest clean traces (non-parametric density proxy)."""
    name = "knn"

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, Z):
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(Z)
        return self

    def score(self, Z):
        d, _ = self.nn_.kneighbors(Z)
        return d.mean(axis=1)


class LOFDetector(BaseDetector):
    """Local Outlier Factor (novelty mode): catches points that are sparse *relative to
    their neighbourhood*, e.g. between the two clean modes."""
    name = "lof"

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, Z):
        self.lof_ = LocalOutlierFactor(n_neighbors=self.k, novelty=True).fit(Z)
        return self

    def score(self, Z):
        return -self.lof_.score_samples(Z)


class IForestDetector(BaseDetector):
    """Isolation Forest on whitened features (rotation-robust variant of the usual IF)."""
    name = "iforest"

    def __init__(self, n_estimators: int = 400, seed: int = 0):
        self.n_estimators, self.seed = n_estimators, seed

    def fit(self, Z):
        self.if_ = IsolationForest(n_estimators=self.n_estimators, max_samples=256,
                                   random_state=self.seed, n_jobs=1).fit(Z)
        return self

    def score(self, Z):
        return -self.if_.score_samples(Z)


class OCSVMDetector(BaseDetector):
    """RBF one-class SVM: a smooth boundary around the clean support."""
    name = "ocsvm"

    def __init__(self, nu: float = 0.05):
        self.nu = nu

    def fit(self, Z):
        self.svm_ = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu).fit(Z)
        return self

    def score(self, Z):
        return -self.svm_.decision_function(Z)


class MaxAxisDetector(BaseDetector):
    """Largest absolute whitened coordinate (a robust 'any component off' test).  Cheap
    complement to the quadratic-form Mahalanobis score: it is sensitive to a single
    large deviation along one decorrelated direction."""
    name = "maxaxis"

    def fit(self, Z):
        return self

    def score(self, Z):
        return np.abs(Z).max(axis=1)


REGISTRY = {
    "mahalanobis": MahalanobisDetector,
    "gmm": GMMDetector,
    "knn": KNNDetector,
    "lof": LOFDetector,
    "iforest": IForestDetector,
    "ocsvm": OCSVMDetector,
    "maxaxis": MaxAxisDetector,
}


def build(names: List[str], seed: int = 0) -> Dict[str, BaseDetector]:
    out = {}
    for n in names:
        cls = REGISTRY[n]
        try:
            out[n] = cls(seed=seed)
        except TypeError:
            out[n] = cls()
    return out
