import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from hpcdet import HPCBackdoorDetector
from hpcdet.calibration import TailCalibrator
from hpcdet.data import DEFAULT_COLUMNS, load_traces


def synth_clean(n, seed):
    """Two-regime clean traces resembling the provided file (log-normal, correlated)."""
    r = np.random.default_rng(seed)
    comp = r.random(n) < 0.7
    base = np.where(comp[:, None], [np.log(1.85e8), np.log(4.78e9), np.log(6.73e6)],
                    [np.log(1.755e8), np.log(4.75e9), np.log(6.26e6)])
    D = np.diag([0.014, 0.005, 0.02])
    cov = D @ np.array([[1.0, 0.2, 0.7], [0.2, 1.0, 0.2], [0.7, 0.2, 1.0]]) @ D
    Z = r.multivariate_normal(np.zeros(3), cov, size=n)
    return pd.DataFrame(np.exp(base + Z), columns=list(DEFAULT_COLUMNS))


class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synth_clean(800, 0)
        cls.det = HPCBackdoorDetector(n_repeats=1, seed=0).fit(cls.train)

    def test_clean_fpr_close_to_target(self):
        fpr = self.det.predict(synth_clean(4000, 1)).mean()
        self.assertLess(abs(fpr - 0.05), 0.025)

    def test_detects_large_shift(self):
        t = synth_clean(500, 2)
        t["cycles"] *= 1.03      # +3% cycles ~ 6 sigma
        self.assertGreater(self.det.predict(t).mean(), 0.9)

    def test_detects_broken_ratio(self):
        t = synth_clean(500, 3)
        t["LLC-loads"] *= 1.10
        self.assertGreater(self.det.predict(t).mean(), 0.9)

    def test_invalid_rows_flagged_not_crash(self):
        t = synth_clean(5, 4)
        t.iloc[0, 0] = np.nan
        t.iloc[1, 1] = 0.0
        t.iloc[2, 2] = -5.0
        pred = self.det.predict(t)
        self.assertTrue(pred[:3].all())

    def test_column_order_and_extra_columns(self):
        t = synth_clean(50, 5)
        t2 = t[["LLC-loads", "cycles", "cache-references"]].copy()
        t2["label"] = 0
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            t2.to_csv(p, index=False)
            X = load_traces(p)
        np.testing.assert_allclose(self.det.score(X), self.det.score(t))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.joblib")
            self.det.save(p)
            d2 = HPCBackdoorDetector.load(p)
        t = synth_clean(100, 6)
        np.testing.assert_allclose(self.det.score(t), d2.score(t))
        self.assertEqual(self.det.threshold_, d2.threshold_)

    def test_calibrator_monotone_and_extrapolates(self):
        ref = np.random.default_rng(0).gamma(2, size=1000)
        c = TailCalibrator().fit(ref)
        s = np.linspace(0, 40, 400)
        v = c.log_surv(s)
        self.assertTrue((np.diff(v) >= -1e-12).all())
        self.assertGreater(v[-1], v[np.searchsorted(s, ref.max())])   # keeps ranking beyond sample max


if __name__ == "__main__":
    unittest.main()
