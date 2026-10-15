"""Synthetic 'backdoor' perturbations used ONLY for model selection and sanity checks.

No real backdoor trace is available (strict anomaly-detection setting), so we cannot know how
a trigger changes the counters.  Instead of guessing one perturbation (as a single hand-picked
"llc x1.3" rule would), we evaluate against a *family* of plausible effects and prefer
detectors/ensembles that are good on all of them.  Magnitudes are expressed in units of the
clean standard deviation so they are meaningful regardless of counter scale.

All perturbations act on log-counters (i.e. multiplicative changes of the raw counts).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

COLS = {"cache-references": 0, "cycles": 1, "LLC-loads": 2}


class Perturber:
    def __init__(self, Xlog_clean: np.ndarray):
        self.mu = Xlog_clean.mean(axis=0)
        self.cov = np.cov(Xlog_clean, rowvar=False)
        self.sd = np.sqrt(np.diag(self.cov))
        vals, vecs = np.linalg.eigh(self.cov)
        self.vals, self.vecs = vals, vecs                   # ascending eigenvalues

    # each family: f(X, m, rng) -> X'
    def families(self) -> Dict[str, Callable]:
        mu, sd, vals, vecs = self.mu, self.sd, self.vals, self.vecs

        def shift(vec):
            return lambda X, m, rng: X + m * vec

        def rand_dir(X, m, rng):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            return X + m * (vecs * np.sqrt(vals)) @ u        # m whitened-sigmas, random direction

        def var_inflate(X, m, rng):
            return mu + (1 + 0.5 * m) * (X - mu)

        def decorrelate(X, m, rng):
            return np.column_stack([rng.permutation(X[:, j]) for j in range(3)])

        def add_noise(X, m, rng):
            return X + rng.normal(size=X.shape) * (0.5 * m) * np.sqrt(vals.mean())

        def unit(j):
            e = np.zeros(3); e[j] = sd[j]; return e

        llc_cr = unit(0) + unit(2)                           # LLC and cache-ref up together
        return {
            "random_direction": rand_dir,
            "cycles_up": shift(unit(1)),
            "cycles_down": shift(-unit(1)),
            "cache_ref_up": shift(unit(0)),
            "llc_up (miss-ratio change)": shift(unit(2)),
            "llc+cacheref_up (more memory traffic)": shift(llc_cr),
            "minor_axis (breaks correlation)": shift(np.sqrt(vals[0]) * vecs[:, 0]),
            "major_axis": shift(np.sqrt(vals[-1]) * vecs[:, -1]),
            "variance_inflation": var_inflate,
            "decorrelated_marginals": decorrelate,
            "extra_noise": add_noise,
        }
