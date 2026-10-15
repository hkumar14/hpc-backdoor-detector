#!/usr/bin/env python
"""One-command demonstration for beginners.

    python demo.py

What it does (takes well under a minute):
  1. Reads the 800 clean traces in data/trace.csv.
  2. Hides 20% of them (160 traces) that the detector never sees during training.
  3. Trains a detector on the other 640 clean traces.
  4. Builds a mock "private validation file": the 160 hidden clean traces plus 160 FAKE backdoor
     traces made by changing the counters of other hidden traces (4 kinds of change).
  5. Writes that file to data/mock_validation.csv and scores it, printing the same metrics the
     organisers will use: Accuracy, TPR, FPR, F1, AUROC.

The fake backdoors are only an imitation (nobody knows what the real ones look like), so treat the
numbers as a sanity check that everything is installed and working, not as a prediction of your rank.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from hpcdet import HPCBackdoorDetector
from hpcdet.data import load_traces
from hpcdet.simulate import Perturber

X = load_traces("data/trace.csv")
rng = np.random.default_rng(7)
order = rng.permutation(len(X))
train_idx, hold_idx = order[:640], order[640:]

print(f"Training on {len(train_idx)} clean traces (this takes ~15-60 s) ...")
det = HPCBackdoorDetector(seed=0).fit(X.iloc[train_idx])

Xlog = np.log(X.values)
families = Perturber(Xlog[train_idx]).families()
kinds = ["random_direction", "llc+cacheref_up (more memory traffic)", "cycles_up", "variance_inflation"]
fake = []
for i, name in enumerate(kinds):
    part = hold_idx[i::4]
    fake.append(np.exp(families[name](Xlog[part], 2.5, rng)))
fake = np.vstack(fake)
clean = X.values[hold_idx]

val = pd.DataFrame(np.vstack([clean, fake]), columns=X.columns)
val["label"] = np.r_[np.zeros(len(clean)), np.ones(len(fake))].astype(int)
val = val.sample(frac=1, random_state=1).reset_index(drop=True)
val.to_csv("data/mock_validation.csv", index=False)
print(f"Wrote data/mock_validation.csv: {len(val)} rows ({int(val.label.sum())} fake backdoors)")

score = det.score(val[list(X.columns)])
pred = (score > det.threshold_).astype(int)
y = val["label"].values
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
print("\nConfusion matrix (rows = truth, columns = prediction)")
print(f"                 predicted clean   predicted backdoor")
print(f"  truly clean        {tn:5d}              {fp:5d}")
print(f"  truly backdoor     {fn:5d}              {tp:5d}")
print(f"\nAccuracy {accuracy_score(y, pred):.3f} | TPR {tp / (tp + fn):.3f} | FPR {fp / (fp + tn):.3f} | "
      f"F1 {f1_score(y, pred):.3f} | AUROC {roc_auc_score(y, score):.3f}")
