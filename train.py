#!/usr/bin/env python
"""Fit the detector on clean HPC traces and save it.

    python train.py --data data/trace.csv --out models/detector.joblib
"""
import argparse
import logging

import numpy as np

from hpcdet import HPCBackdoorDetector
from hpcdet.data import load_traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/trace.csv")
    ap.add_argument("--out", default="models/detector.joblib")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="fraction of clean traces allowed to be flagged (threshold calibration)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    X = load_traces(args.data)
    det = HPCBackdoorDetector(target_fpr=args.target_fpr, seed=args.seed).fit(X)
    det.save(args.out)

    s = det.score(X)
    print(f"trained on {det.n_train_} clean traces | detectors={det.detector_names} "
          f"| GMM components={det.models_['gmm'].n_components_}")
    print(f"threshold (fused evidence) = {det.threshold_:.4f}  (target clean FPR {args.target_fpr:.0%})")
    print(f"in-sample flagged fraction = {(s > det.threshold_).mean():.3%}  "
          f"(out-of-fold flagged fraction = {(det.oof_fused_ > det.threshold_).mean():.3%})")
    print(f"saved -> {args.out}  and  {args.out.rsplit('.', 1)[0]}.json")


if __name__ == "__main__":
    main()
