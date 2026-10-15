"""End-to-end detector: log-transform -> whiten -> {one-class detectors} -> tail calibration
-> evidence fusion -> threshold at a target clean FPR (estimated from out-of-fold scores)."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import detectors as D
from .calibration import TailCalibrator
from .data import DEFAULT_COLUMNS

log = logging.getLogger(__name__)

DEFAULT_DETECTORS = ("gmm", "knn", "iforest")


def _to_log(X, columns) -> np.ndarray:
    """Raw counts -> log.  Invalid rows (NaN / <=0) become NaN and are flagged by score()."""
    if isinstance(X, pd.DataFrame):
        X = X[list(columns)].to_numpy(dtype=float)
    X = np.asarray(X, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(X > 0, np.log(X), np.nan)
    return out


def fuse(C: np.ndarray, how: str = "mean") -> np.ndarray:
    """Combine per-detector calibrated evidence (columns of C)."""
    if how == "mean":
        return C.mean(axis=1)
    if how == "max":
        return C.max(axis=1)
    if how == "median":
        return np.median(C, axis=1)
    raise ValueError(how)


class HPCBackdoorDetector:
    """Fit on clean traces only.  ``score`` is larger for more suspicious traces and
    ``predict`` returns 1 for backdoor/anomalous, 0 for clean."""

    def __init__(self, detectors: Sequence[str] = DEFAULT_DETECTORS, fusion: str = "mean",
                 target_fpr: float = 0.05, n_splits: int = 5, n_repeats: int = 3,
                 columns: Sequence[str] = DEFAULT_COLUMNS, seed: int = 0):
        self.detector_names = list(detectors)
        self.fusion = fusion
        self.target_fpr = target_fpr
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.columns = list(columns)
        self.seed = seed

    # ------------------------------------------------------------------ fitting
    def _fit_models(self, Xlog: np.ndarray):
        wh = D.Whitener().fit(Xlog)
        Z = wh.transform(Xlog)
        dets = D.build(self.detector_names, seed=self.seed)
        for d in dets.values():
            d.fit(Z)
        return wh, dets

    @staticmethod
    def _raw_scores(wh, dets, Xlog: np.ndarray) -> Dict[str, np.ndarray]:
        Z = wh.transform(Xlog)
        return {n: d.score(Z) for n, d in dets.items()}

    def fit(self, X) -> "HPCBackdoorDetector":
        Xlog = _to_log(X, self.columns)
        keep = np.isfinite(Xlog).all(axis=1)
        if not keep.all():
            log.warning("dropping %d invalid training rows", int((~keep).sum()))
        Xlog = Xlog[keep]
        n = len(Xlog)
        if n < 50:
            raise ValueError("need at least 50 clean traces")

        # 1) out-of-fold clean scores: honest estimate of how clean *unseen* traces score
        oof = {name: [] for name in self.detector_names}
        for r in range(self.n_repeats):
            kf = KFold(self.n_splits, shuffle=True, random_state=self.seed + 1000 * r)
            for tr, te in kf.split(Xlog):
                wh, dets = self._fit_models(Xlog[tr])
                sc = self._raw_scores(wh, dets, Xlog[te])
                for k, v in sc.items():
                    oof[k].append(v)
        self.calibrators_ = {k: TailCalibrator().fit(np.concatenate(v)) for k, v in oof.items()}

        # 2) fused OOF evidence -> threshold at the requested clean FPR
        C_oof = np.column_stack([self.calibrators_[k].log_surv(np.concatenate(oof[k]))
                                 for k in self.detector_names])
        fused_oof = fuse(C_oof, self.fusion)
        self.oof_fused_ = fused_oof
        self.threshold_ = float(np.quantile(fused_oof, 1.0 - self.target_fpr))

        # 3) final models on all clean data
        self.whitener_, self.models_ = self._fit_models(Xlog)
        self.n_train_ = n
        return self

    # ---------------------------------------------------------------- inference
    def calibrated_scores(self, X) -> np.ndarray:
        """(n, n_detectors) matrix of calibrated evidence; invalid rows -> +inf."""
        Xlog = _to_log(X, self.columns)
        ok = np.isfinite(Xlog).all(axis=1)
        C = np.full((len(Xlog), len(self.detector_names)), np.inf)
        if ok.any():
            raw = self._raw_scores(self.whitener_, self.models_, Xlog[ok])
            C[ok] = np.column_stack([self.calibrators_[k].log_surv(raw[k])
                                     for k in self.detector_names])
        return C

    def score(self, X) -> np.ndarray:
        return fuse(self.calibrated_scores(X), self.fusion)

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold_ if threshold is None else threshold
        return (self.score(X) > t).astype(int)

    def set_target_fpr(self, fpr: float) -> None:
        self.target_fpr = fpr
        self.threshold_ = float(np.quantile(self.oof_fused_, 1.0 - fpr))

    # --------------------------------------------------------------------- I/O
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        meta = {
            "detectors": self.detector_names, "fusion": self.fusion,
            "target_fpr": self.target_fpr, "threshold": self.threshold_,
            "columns": self.columns, "n_train": self.n_train_,
            "gmm_components": getattr(self.models_.get("gmm"), "n_components_", None),
        }
        with open(path.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path: str) -> "HPCBackdoorDetector":
        return joblib.load(path)
