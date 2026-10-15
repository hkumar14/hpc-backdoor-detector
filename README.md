# Detection of neural-backdoor inferences from Hardware Performance Counters

Strict **one-class anomaly detection**: the detector is trained on the 800 *clean* HPC traces only
(`cache-references`, `cycles`, `LLC-loads`) and labels unseen traces as clean (0) or backdoor (1).

Short report: [`reports/report.pdf`](reports/report.pdf)

## Quick start

```bash
pip install -r requirements.txt

python train.py                                   # fits on data/trace.csv -> models/detector.joblib (+ .json)
python predict.py --input my_traces.csv --output predictions.csv
```

`predict.py` writes `score` (higher = more suspicious) and `prediction` (1 = backdoor). If the
input also has a label column (`label`, `is_backdoor`, `y`, `target` or `backdoor`; 1 = backdoor) it
prints **Accuracy, TPR, FPR, F1 and AUROC**. Columns are matched by name, so order and extra columns
do not matter. Rows with NaN / zero / negative counters are returned as anomalous instead of crashing.

Change the operating point without retraining: `python predict.py --input f.csv --target-fpr 0.02`.

Use from Python:

```python
from hpcdet import HPCBackdoorDetector
from hpcdet.data import load_traces

det = HPCBackdoorDetector.load("models/detector.joblib")
X = load_traces("my_traces.csv")
scores, labels = det.score(X), det.predict(X)
```

## Method in one paragraph

log-counters -> winsorised whitening -> **three one-class models fitted on clean data**
(Gaussian-mixture likelihood with BIC-chosen components, kNN distance, Isolation Forest) ->
each score is mapped to `-log P_clean(S >= s)` using **repeated 5-fold out-of-fold clean scores**
(empirical bulk + exponential tail) -> evidence is averaged -> the threshold is the 95th percentile of
the fused out-of-fold clean evidence, i.e. **clean FPR is controlled at 5% by construction** (the
realised held-out FPR is 4.4%). See the report for the data analysis behind each choice.

## Why not the "Random Forest on clean data" idea?

A classifier trained on a single class has nothing to separate: its output is constant (AUROC exactly 0.5,
reproduced in `evaluate.py`). Isolation Forest on the raw 9 engineered features also trails the final
ensemble (AUROC 0.846 vs 0.867 in the benchmark below).

## Validation (read this honestly)

No backdoor trace is provided, so **real TPR cannot be measured**. `evaluate.py` runs a nested 5-fold
benchmark where the held-out clean fold is compared with perturbed copies of itself under 11 families
of plausible trigger effects (single-counter shifts, miss-ratio change, more memory traffic, shifts
along principal axes, correlation breaking, variance inflation, extra noise) at 1/2/3 clean standard
deviations. This is a relative yardstick for choosing methods and the threshold, not a forecast of the
private-set score.

| Method | mean AUROC | worst-family AUROC |
|---|---|---|
| **Submitted ensemble (GMM + kNN + IForest)** | 0.867 | 0.788 |
| GMM only | 0.869 | 0.797 |
| kNN only | 0.866 | 0.790 |
| Isolation Forest only | 0.853 | 0.759 |
| Prior plan: IForest on 9 raw/ratio/log features | 0.846 | 0.761 |
| Prior plan: single-class Random Forest | 0.500 | 0.500 |

Realised FPR on held-out clean data tracks the target (target 5% -> 4.4%). For effects of 1 / 2 / 3
clean standard deviations the AUROC is about 0.76 / 0.89 / 0.95.

## Repository layout

```
hpcdet/            data.py (loading), detectors.py (models), calibration.py, pipeline.py, simulate.py, benchmark.py
train.py           fit + save detector
predict.py         score a CSV, optional metrics
evaluate.py        reproduce every number in the report (~8 min on 2 cores)
make_report.py     rebuild figures and reports/report.pdf
models/            detector.joblib (saved model) and detector.json (threshold and metadata)
reports/           report.pdf, figures, benchmark CSVs
tests/             unit tests: python -m unittest discover -s tests
data/trace.csv     the provided clean traces
```

All randomness is seeded, so `train.py` is deterministic. Tested with Python 3.13, scikit-learn 1.9, numpy 2.5.
