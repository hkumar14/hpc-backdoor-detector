#!/usr/bin/env python
"""Score a CSV of HPC traces.

    python predict.py --input traces.csv --output predictions.csv

Output columns: score (higher = more suspicious), prediction (1 = backdoor, 0 = clean).
If the input has a label column (label / is_backdoor / y / target; 1 = backdoor) the usual
metrics (Accuracy, TPR, FPR, F1, AUROC) are printed as well.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from hpcdet import HPCBackdoorDetector
from hpcdet.data import load_traces

LABEL_NAMES = ("label", "is_backdoor", "y", "target", "backdoor")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--model", default="models/detector.joblib")
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="optionally re-pick the operating point (e.g. 0.02) without retraining")
    args = ap.parse_args()

    det = HPCBackdoorDetector.load(args.model)
    if args.target_fpr is not None:
        det.set_target_fpr(args.target_fpr)
    X = load_traces(args.input, det.columns)
    score = det.score(X)
    pred = (score > det.threshold_).astype(int)
    out = pd.DataFrame({"score": score, "prediction": pred})
    out.to_csv(args.output, index=False)
    print(f"{len(out)} traces scored, {int(pred.sum())} flagged as backdoor -> {args.output}")

    raw = pd.read_csv(args.input)
    lab = next((c for c in raw.columns if c.strip().lower() in LABEL_NAMES), None)
    if lab is not None:
        y = raw[lab].astype(int).values
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        finite = np.isfinite(score)
        auc = roc_auc_score(y[finite], score[finite]) if len(set(y)) > 1 else float("nan")
        print(f"Accuracy {accuracy_score(y, pred):.4f} | TPR {tp / max(tp + fn, 1):.4f} | "
              f"FPR {fp / max(fp + tn, 1):.4f} | F1 {f1_score(y, pred):.4f} | AUROC {auc:.4f}")


if __name__ == "__main__":
    main()
