#!/usr/bin/env python
"""Build figures and the short PDF report (reports/report.pdf) from the saved model and the
CSV files written by evaluate.py.

    python make_report.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from hpcdet import HPCBackdoorDetector
from hpcdet.data import load_traces
from hpcdet.simulate import Perturber

R = "reports"
F = f"{R}/figures"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"

X = load_traces("data/trace.csv")
Xlog = np.log(X.values)
det = HPCBackdoorDetector.load("models/detector.joblib")
bench = pd.read_csv(f"{R}/benchmark_detectors.csv")
op = pd.read_csv(f"{R}/operating_points.csv")
SUB = "ENSEMBLE (gmm+knn+iforest)  [submitted]"

# ---------------------------------------------------------------- figure 1: the data
Z = det.whitener_.transform(Xlog)
comp = det.models_["gmm"].gmm_.predict(Z)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
pal = [BLUE, ORANGE, "#38a169", "#805ad5"]
for k in np.unique(comp):
    m = comp == k
    ax[0].scatter(X.values[m, 0] / 1e6, X.values[m, 2] / 1e6, s=6, color=pal[k % 4], alpha=.7)
    ax[1].scatter(X.values[m, 0] / 1e6, X.values[m, 1] / 1e9, s=6, color=pal[k % 4], alpha=.7)
ax[0].set(xlabel="cache-references (M)", ylabel="LLC-loads (M)", title="two operating regimes, r = 0.92")
ax[1].set(xlabel="cache-references (M)", ylabel="cycles (G)", title="cycles: rare spikes")
ax[2].hist(X["cycles"] / 1e9, bins=60, color=BLUE)
ax[2].set(xlabel="cycles (G)", ylabel="count", title="cycles distribution (2 spikes > 4.9 G)")
plt.tight_layout(); plt.savefig(f"{F}/fig1_data.png", dpi=170); plt.close()

# ----------------------------------------- figure 2: held-out scores, clean vs simulated backdoor
rng = np.random.default_rng(3)
idx = rng.permutation(len(X))
tr, te = idx[:640], idx[640:]
d80 = HPCBackdoorDetector(seed=3).fit(X.iloc[tr])
fams = Perturber(Xlog[tr]).families()
sc_clean = d80.score(X.iloc[te])
sc_bd = np.concatenate([d80.score(pd.DataFrame(np.exp(f(Xlog[te], 2.0, rng)), columns=X.columns))
                        for f in fams.values()])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bins = np.linspace(0, max(np.quantile(sc_bd, .99), 12), 60)
ax.hist(sc_clean, bins=bins, alpha=.75, color=BLUE, density=True, label="held-out clean")
ax.hist(sc_bd, bins=bins, alpha=.6, color=ORANGE, density=True, label="simulated backdoor (+2 sigma, all 11 families)")
ax.axvline(d80.threshold_, color="k", ls="--", lw=1); ax.text(d80.threshold_ + .1, ax.get_ylim()[1] * .85, "threshold\n(5% clean FPR)", fontsize=8)
ax.set(xlabel="fused anomaly evidence  (-log clean tail probability)", ylabel="density"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{F}/fig2_scores.png", dpi=170); plt.close()

# ----------------------------------------------- figure 3: AUROC heat-map, methods x attack family
b2 = bench[bench.magnitude == 2.0]
meths = [SUB, "gmm", "knn", "iforest", "mahalanobis",
         "baseline: IsolationForest on 9 raw/ratio/log features", "baseline: single-class RandomForest"]
short = {SUB: "Submitted ensemble", "gmm": "GMM only", "knn": "kNN only", "iforest": "IForest only",
         "mahalanobis": "Mahalanobis only",
         "baseline: IsolationForest on 9 raw/ratio/log features": "Prior plan: IForest (9 feats)",
         "baseline: single-class RandomForest": "Prior plan: 1-class RF"}
H = b2[b2.method.isin(meths)].groupby(["method", "family"]).auroc.mean().unstack(0)[meths]
fig, ax = plt.subplots(figsize=(8.2, 3.9))
im = ax.imshow(H.values, cmap="Blues", vmin=.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(meths)), [short[m] for m in meths], rotation=28, ha="right")
ax.set_yticks(range(len(H)), H.index)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if H.values[i, j] > .8 else "black")
ax.spines[:].set_visible(False); plt.colorbar(im, label="AUROC", fraction=.035)
ax.set_title("AUROC at a +2 sigma simulated backdoor effect (nested 5-fold CV)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{F}/fig3_heatmap.png", dpi=170); plt.close()

# ------------------------------------------------------------ figure 4: operating points
g = op.groupby(["magnitude", "target_fpr"])[["fpr", "tpr"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for m, c in zip((1.0, 2.0, 3.0), (GREY, BLUE, ORANGE)):
    s = g[g.magnitude == m]
    ax.plot(s.fpr, s.tpr, "-o", color=c, ms=4, label=f"effect = {m:.0f} sigma")
ax.set(xlabel="realised FPR on held-out clean traces", ylabel="TPR on simulated backdoors", xlim=(0, .11), ylim=(0, 1))
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(f"{F}/fig4_operating.png", dpi=170); plt.close()

# ------------------------------------------------------------------------------ numbers
mean_auc = bench.groupby("method").auroc.mean()
worst_auc = bench.groupby(["method", "family"]).auroc.mean().groupby("method").min()
opm = op.groupby("target_fpr")[["fpr", "tpr", "accuracy", "f1"]].mean()
op23 = op[op.magnitude >= 2].groupby("target_fpr")[["fpr", "tpr", "accuracy", "f1"]].mean()
r5 = opm.loc[0.05]; r5b = op23.loc[0.05]
gmm_k = det.models_["gmm"].n_components_

# --------------------------------------------------------------------------------- PDF
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.4, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=3)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#4a5568"))
P = lambda t, s=body: Paragraph(t, s)


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.3), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#cbd5e0")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t


doc = SimpleDocTemplate(f"{R}/report.pdf", pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Detecting neural backdoors from HPC traces",
                        author="HPC backdoor detector")
S = []
S += [P("Detecting neural-network backdoor inferences from hardware performance counters", h1),
      P("Strict one-class anomaly detection &mdash; ensemble of density models with out-of-fold calibration", small), Spacer(1, 6)]

S += [P("1. Problem and what the data allows", h2),
      P("We receive 800 HPC traces (cache-references, cycles, LLC-loads) from <b>clean</b> inferences only and must label unseen "
        "traces as clean or backdoor. No backdoor example exists, so any supervised classifier is impossible; the problem is "
        "novelty detection. The private set will be scored with Accuracy, TPR, FPR, F1 and AUROC, so we need <i>both</i> a good ranking "
        "(AUROC) and a well-calibrated decision threshold."),
      P("Findings from the clean file (Fig. 1): (i) the counters are strictly positive and multiplicative, so we model "
        "<b>log-counters</b> (a ratio such as LLC/cache-ref becomes a difference); (ii) cache-references and LLC-loads are highly "
        f"correlated (r&nbsp;=&nbsp;0.92) and the data form <b>multiple operating regimes</b> (BIC picks {gmm_k} Gaussian components; a single "
        "Gaussian is the wrong shape); (iii) rows are independent (lag-1 autocorrelation &asymp; 0), so no temporal model is needed; "
        "(iv) cycles has rare OS-noise spikes (two rows at 6&ndash;9 robust sigma), which must not be learned as &ldquo;normal&rdquo; nor "
        "cause a flood of false positives."),
      Image(f"{F}/fig1_data.png", width=17.2 * cm, height=17.2 * cm * 3.1 / 10.5)]

S += [P("2. Method", h2),
      P("<b>Pipeline.</b> raw counts &rarr; log &rarr; winsorised whitening (ZCA) &rarr; three one-class detectors fitted on clean data only "
        "&rarr; tail calibration &rarr; evidence fusion &rarr; threshold."),
      table([[P("<b>Detector</b>", small), P("<b>Models</b>", small), P("<b>Why it is in the ensemble</b>", small)],
             [P("GMM (BIC-selected k)", small), P("negative log-likelihood of the clean density", small), P("matches the multi-regime shape; best single model", small)],
             [P("kNN (k = 10)", small), P("mean distance to nearest clean traces", small), P("non-parametric; no distributional assumption", small)],
             [P("Isolation Forest", small), P("random-partition isolation depth on whitened features", small), P("independent inductive bias; axis-aligned view", small)]],
            [3.6 * cm, 6.4 * cm, 7.2 * cm]),
      Spacer(1, 4),
      P("<b>Calibration (the part that fixes accuracy / F1 / FPR).</b> Scores from different detectors are not comparable, and "
        "in-sample scores of the training rows are over-optimistic. We therefore compute <i>repeated 5-fold out-of-fold</i> clean scores "
        "(3 repeats) and, per detector, map any score <i>s</i> to <i>&minus;log P<sub>clean</sub>(S&ge;s)</i>: empirical survival "
        "function in the bulk and an exponential (GPD, &xi;=0) tail above the 90th percentile, so very strong anomalies keep ranking above "
        "moderate ones instead of saturating at 1/n. The three evidence values are averaged (a Fisher-style fusion that rewards any detector "
        "that sees something). The decision threshold is the 95th percentile of the fused <i>out-of-fold</i> clean evidence, so the clean-FPR is "
        "controlled <i>by construction</i> at 5% and does not depend on a guessed contamination rate. Final models are then refit on all 800 traces."),
      P("<b>Robustness details.</b> columns are selected by name (order and extra columns such as a label are ignored); rows with "
        "NaN/zero/negative counters cannot be modelled and are returned as anomalous instead of crashing; whitening is winsorised "
        "(1%/99%) so the two cycle spikes do not distort the metric; everything is seeded and covered by unit tests.")]

S += [P("3. Validation protocol &mdash; and its honest limit", h2),
      P("Because there are no backdoor traces, <b>we cannot measure real TPR</b>. Rather than hand-picking a single synthetic anomaly "
        "(e.g. &ldquo;LLC&times;1.3&rdquo;), we benchmark against <b>11 families of plausible trigger effects</b> at 1, 2 and 3 clean "
        "standard deviations: single-counter shifts up/down, miss-ratio change (LLC only), more memory traffic (LLC and cache-ref together), "
        "shifts along the major and minor principal axes (the minor axis breaks the correlation structure while keeping marginals "
        "normal), random directions, variance inflation, independent column shuffling (marginals intact, dependence destroyed) and "
        "extra noise. A <b>nested</b> 5-fold protocol is used: the entire pipeline, including its inner out-of-fold calibration and "
        "threshold, is fitted on 4/5 of the clean data and tested on the held-out clean fifth (negatives) versus a perturbed copy (positives). "
        "These numbers are a relative yardstick for choosing methods and thresholds, not a forecast of the private-set score."),
      Image(f"{F}/fig3_heatmap.png", width=14.6 * cm, height=14.6 * cm * 3.9 / 8.2)]

rows = [["Method", "mean AUROC", "worst-family AUROC", "TPR @ 5% FPR"]]
order = [SUB, "gmm", "knn", "lof", "iforest", "mahalanobis", "ocsvm",
         "baseline: IsolationForest on 9 raw/ratio/log features", "baseline: single-class RandomForest"]
nice = {SUB: "<b>Submitted ensemble (GMM+kNN+IForest)</b>", "gmm": "GMM only", "knn": "kNN only", "lof": "LOF only",
        "iforest": "Isolation Forest only", "mahalanobis": "Robust Mahalanobis only", "ocsvm": "One-class SVM only",
        "baseline: IsolationForest on 9 raw/ratio/log features": "Prior plan: IForest on 9 raw/ratio/log features",
        "baseline: single-class RandomForest": "Prior plan: Random Forest trained on one class"}
tpr = bench.groupby("method").tpr_at_5fpr.mean()
for m in order:
    rows.append([P(nice[m], small), f"{mean_auc[m]:.3f}", f"{worst_auc[m]:.3f}", f"{tpr[m]:.3f}"])
mt = bench[bench.family.str.startswith("llc+cache")].groupby("method").auroc.mean()
mh, kn = mt["mahalanobis"], mt["knn"]
S += [Spacer(1, 3), table(rows, [8.2 * cm, 2.6 * cm, 3.4 * cm, 3 * cm]), Spacer(1, 3),
      P(f"Mean over 11 families &times; 3 magnitudes &times; 5 folds. The best single model (GMM) and the ensemble are statistically tied, on both "
        "mean and worst-family AUROC. The ensemble is kept as insurance: the per-family winner among GMM and kNN changes (GMM wins 6 families, kNN 5), "
        f"and a single model can fall behind on an effect we did not simulate (e.g. Mahalanobis reaches only {mh:.2f} on &ldquo;more memory traffic&rdquo; where kNN "
        f"reaches {kn:.2f}). The single-class Random "
        "Forest in the earlier plan outputs a constant score (AUROC exactly 0.5): a classifier cannot flag a class it has never seen.", small),
      PageBreak()]

S += [P("4. Operating point", h2),
      P("The fused-evidence threshold is set so that the realised clean FPR equals the target. The check below is on held-out clean folds "
        "with the full nested pipeline: realised FPR tracks the target (slightly conservative), and accuracy / F1 flatten out beyond "
        "about 5&ndash;8% FPR, so we deploy <b>5%</b>. If the private set contains a much smaller share of backdoor traces, lower it "
        "(<font face='Courier'>predict.py --target-fpr 0.02</font>); no retraining is needed."),
      table([["target FPR", "realised FPR", "TPR (all effects)", "TPR (>= 2 sigma)", "Acc (50/50, >= 2 sigma)", "F1 (>= 2 sigma)"]] +
            [[f"{a:.0%}", f"{opm.loc[a, 'fpr']:.3f}", f"{opm.loc[a, 'tpr']:.3f}", f"{op23.loc[a, 'tpr']:.3f}",
              f"{op23.loc[a, 'accuracy']:.3f}", f"{op23.loc[a, 'f1']:.3f}"] for a in opm.index],
            [2.2 * cm, 2.6 * cm, 3 * cm, 3 * cm, 3.7 * cm, 2.7 * cm]),
      Spacer(1, 4),
      Table([[Image(f"{F}/fig2_scores.png", width=8.9 * cm, height=8.9 * cm * 3.0 / 6.4),
              Image(f"{F}/fig4_operating.png", width=7.6 * cm, height=7.6 * cm * 3.3 / 5.2)]])]

S += [P("5. Limitations and how to use", h2),
      P("&bull; Real backdoor traces may differ from our simulated families; effects below ~1 sigma in all counters are fundamentally "
        "undetectable from three tightly clustered counters (AUROC &asymp; 0.75 at 1 sigma). Strong effects (&ge;3 sigma) are detected with AUROC &asymp; 0.95."),
      P("&bull; If the private file contains extra HPC columns, the detector uses only the three it was trained on; to use more counters, "
        "retrain with <font face='Courier'>HPCBackdoorDetector(columns=[...])</font>."),
      P("&bull; Commands: <font face='Courier'>python train.py</font> (fits and saves models/detector.joblib), "
        "<font face='Courier'>python predict.py --input file.csv</font> (scores; prints Accuracy/TPR/FPR/F1/AUROC if a label column is present), "
        "<font face='Courier'>python evaluate.py</font> (reproduces every number here), <font face='Courier'>python make_report.py</font>.")]

doc.build(S)
print("wrote reports/report.pdf")
