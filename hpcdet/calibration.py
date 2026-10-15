"""Turning raw anomaly scores into comparable, calibrated evidence values.

Different detectors score on different scales (distances, log-likelihoods, SVM margins).
For each detector we keep the distribution of *out-of-fold clean scores* and map any new
score s to  -log P_clean(S >= s):  an empirical survival function in the bulk and an
exponential (GPD, xi=0) tail beyond the 90th percentile, so that very strong anomalies keep
ranking above moderate ones instead of all saturating at 1/n.
"""
from __future__ import annotations

import numpy as np


class TailCalibrator:
    def __init__(self, tail_quantile: float = 0.90):
        self.tail_quantile = tail_quantile

    def fit(self, ref_scores: np.ndarray) -> "TailCalibrator":
        ref = np.sort(np.asarray(ref_scores, dtype=float))
        self.ref_ = ref
        n = len(ref)
        self.u_ = float(np.quantile(ref, self.tail_quantile))
        exc = ref[ref > self.u_] - self.u_
        self.p_u_ = max(len(exc), 1) / (n + 1.0)
        self.beta_ = float(exc.mean()) if len(exc) else 1e-12
        self.beta_ = max(self.beta_, 1e-12)
        return self

    def log_surv(self, s: np.ndarray) -> np.ndarray:
        """Return -log P(S >= s) (>= 0, monotone increasing in s)."""
        s = np.asarray(s, dtype=float)
        n = len(self.ref_)
        # number of reference scores >= s
        k = n - np.searchsorted(self.ref_, s, side="left")
        emp = (k + 1.0) / (n + 1.0)
        tail = self.p_u_ * np.exp(-(s - self.u_) / self.beta_)
        surv = np.where(s > self.u_, np.minimum(tail, emp), emp)
        out = -np.log(np.clip(surv, 1e-300, 1.0))
        out = np.where(np.isnan(s), np.inf, out)
        return out
