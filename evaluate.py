#!/usr/bin/env python
"""Reproduce every number in the report (nested CV against simulated backdoors).

    python evaluate.py            # ~8 minutes on 2 cores, writes reports/*.csv
"""
import argparse

import pandas as pd

from hpcdet import benchmark as B
from hpcdet.data import load_traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trace.csv")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    X = load_traces(args.data)

    combos = {
        "ENSEMBLE (gmm+knn+iforest)  [submitted]": (["gmm", "knn", "iforest"], "mean"),
        "ensemble: all 7, mean": (list(B.ALL), "mean"),
        "ensemble: all 7, max": (list(B.ALL), "max"),
        "ensemble: gmm+mahalanobis": (["gmm", "mahalanobis"], "mean"),
    }
    df = B.run(X, n_outer=5, n_inst=3, combos=combos, seed=args.seed)
    base = B.run_deepseek_baseline(X, seed=args.seed)
    pd.concat([df, base]).to_csv(f"{args.outdir}/benchmark_detectors.csv", index=False)

    op = B.run_operating_points(X, seed=args.seed)
    op.to_csv(f"{args.outdir}/operating_points.csv", index=False)
    print(pd.concat([df, base]).groupby("method")[["auroc", "tpr_at_5fpr"]].mean()
          .sort_values("auroc", ascending=False).round(3))
    print(op.groupby("target_fpr")[["fpr", "tpr", "accuracy", "f1"]].mean().round(3))


if __name__ == "__main__":
    main()
