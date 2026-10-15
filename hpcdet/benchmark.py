"""Nested cross-validated benchmark against simulated backdoor perturbations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from .pipeline import HPCBackdoorDetector, fuse, _to_log
from .simulate import Perturber

ALL = ("gmm", "mahalanobis", "knn", "lof", "iforest", "ocsvm", "maxaxis")


def run(X: pd.DataFrame, magnitudes=(1.0, 2.0, 3.0), n_outer=5, n_inst=4, seed=0,
        detectors=ALL, combos=None):
    """Returns a long DataFrame: method, family, magnitude, fold, auroc, tpr_at_thr, fpr_at_thr.

    Each outer fold fits the *entire* pipeline on 4/5 of the clean data (including its inner
    out-of-fold calibration / threshold) and evaluates on the held-out clean fold (negatives)
    versus a perturbed copy of it (positives).
    """
    Xlog_all = _to_log(X, ["cache-references", "cycles", "LLC-loads"])
    rng = np.random.default_rng(seed)
    rows = []
    kf = KFold(n_outer, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(Xlog_all)):
        X_tr = X.iloc[tr]
        det = HPCBackdoorDetector(detectors=detectors, n_repeats=1, seed=seed + fold).fit(X_tr)
        pert = Perturber(Xlog_all[tr])
        fams = pert.families()
        Xte_log = Xlog_all[te]
        Cneg = det.calibrated_scores(np.exp(Xte_log))
        names = list(det.detector_names)
        # per-detector thresholds at 5% FPR from OOF reference; combos use fused OOF
        methods = {n: ([n], "mean") for n in names}
        for cname, (members, how) in (combos or {}).items():
            methods[cname] = (members, how)
        # fused-oof thresholds for each method (needs oof per detector -> recompute from calibrators)
        for cname, (members, how) in methods.items():
            idx = [names.index(m) for m in members]
            for fam, f in fams.items():
                for m in magnitudes:
                    aucs, tprs = [], []
                    for _ in range(n_inst):
                        Xp = f(Xte_log, m, rng)
                        Cpos = det.calibrated_scores(np.exp(Xp))
                        sn, sp = fuse(Cneg[:, idx], how), fuse(Cpos[:, idx], how)
                        y = np.r_[np.zeros(len(sn)), np.ones(len(sp))]
                        s = np.r_[sn, sp]
                        aucs.append(roc_auc_score(y, s))
                        thr = np.quantile(sn, 0.95)             # oracle 5% FPR on held-out clean
                        tprs.append(float((sp > thr).mean()))
                    rows.append(dict(method=cname, family=fam, magnitude=m, fold=fold,
                                     auroc=float(np.mean(aucs)), tpr_at_5fpr=float(np.mean(tprs))))
    return pd.DataFrame(rows)


def run_deepseek_baseline(X: pd.DataFrame, magnitudes=(1.0, 2.0, 3.0), n_outer=5, n_inst=3, seed=0):
    """Re-implementation of the previously proposed approach: 9 engineered features
    (raw, ratios, logs) -> StandardScaler -> IsolationForest(300).  Also reports the
    'Random Forest trained on a single class' idea, whose score is constant (AUROC = 0.5)."""
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    def feats(A):
        c, y, l = A[:, 0], A[:, 1], A[:, 2]
        return np.column_stack([c, y, l, l / c, y / c, y / l, np.log1p(c), np.log1p(y), np.log1p(l)])

    Xlog_all = _to_log(X, ["cache-references", "cycles", "LLC-loads"])
    rng = np.random.default_rng(seed)
    rows = []
    for fold, (tr, te) in enumerate(KFold(n_outer, shuffle=True, random_state=seed).split(Xlog_all)):
        sc = StandardScaler().fit(feats(np.exp(Xlog_all[tr])))
        iso = IsolationForest(n_estimators=300, random_state=seed).fit(sc.transform(feats(np.exp(Xlog_all[tr]))))
        rf = RandomForestClassifier(n_estimators=100, random_state=seed).fit(
            sc.transform(feats(np.exp(Xlog_all[tr]))), np.zeros(len(tr)))
        fams = Perturber(Xlog_all[tr]).families()
        Xte_log = Xlog_all[te]

        def s_if(Xl): return -iso.score_samples(sc.transform(feats(np.exp(Xl))))
        def s_rf(Xl): return 1.0 - rf.predict_proba(sc.transform(feats(np.exp(Xl))))[:, 0]
        for name, fn in (("baseline: IsolationForest on 9 raw/ratio/log features", s_if),
                         ("baseline: single-class RandomForest", s_rf)):
            sn = fn(Xte_log)
            for fam, f in fams.items():
                for m in magnitudes:
                    aucs, tprs = [], []
                    for _ in range(n_inst):
                        sp = fn(f(Xte_log, m, rng))
                        y = np.r_[np.zeros(len(sn)), np.ones(len(sp))]
                        aucs.append(roc_auc_score(y, np.r_[sn, sp]))
                        thr = np.quantile(sn, 0.95)
                        tprs.append(float((sp > thr).mean()))
                    rows.append(dict(method=name, family=fam, magnitude=m, fold=fold,
                                     auroc=float(np.mean(aucs)), tpr_at_5fpr=float(np.mean(tprs))))
    return pd.DataFrame(rows)


def run_operating_points(X: pd.DataFrame, fprs=(0.01, 0.02, 0.03, 0.05, 0.08, 0.10),
                         magnitudes=(1.0, 2.0, 3.0), n_outer=5, n_inst=3, seed=0, **det_kw):
    """End-to-end check of the *deployed* decision rule: for several target FPRs, the realised
    FPR on held-out clean traces and TPR / accuracy / F1 on simulated backdoors (50/50 mix)."""
    Xlog_all = _to_log(X, ["cache-references", "cycles", "LLC-loads"])
    rng = np.random.default_rng(seed)
    rows = []
    for fold, (tr, te) in enumerate(KFold(n_outer, shuffle=True, random_state=seed).split(Xlog_all)):
        det = HPCBackdoorDetector(seed=seed + fold, **det_kw).fit(X.iloc[tr])
        fams = Perturber(Xlog_all[tr]).families()
        Xte_log = Xlog_all[te]
        sn = det.score(np.exp(Xte_log))
        for fam, f in fams.items():
            for m in magnitudes:
                for _ in range(n_inst):
                    sp = det.score(np.exp(f(Xte_log, m, rng)))
                    for a in fprs:
                        thr = np.quantile(det.oof_fused_, 1 - a)
                        fp = float((sn > thr).mean()); tp = float((sp > thr).mean())
                        n = len(sn)
                        acc = (n * (1 - fp) + n * tp) / (2 * n)
                        prec = tp / max(tp + fp, 1e-12)
                        f1 = 2 * prec * tp / max(prec + tp, 1e-12)
                        rows.append(dict(target_fpr=a, family=fam, magnitude=m, fold=fold,
                                         fpr=fp, tpr=tp, accuracy=acc, f1=f1))
    return pd.DataFrame(rows)
