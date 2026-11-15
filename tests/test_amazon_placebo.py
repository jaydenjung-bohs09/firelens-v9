"""Reproduce the seeded Amazon five-day placebo without changing fire records."""
import json
import unittest
from datetime import date,timedelta
from v15.amazon_tier2 import ROOT
from v15.amazon_placebo import placebo

class AmazonPlaceboTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=json.loads((ROOT/"v15/amazon_placebo_results.json").read_text())
        cls.actual=placebo()

    def test_seeded_nonoverlapping_windows(self):
        self.assertEqual(self.actual["seed"],20261115)
        self.assertEqual(self.actual["window_count"],30)
        self.assertEqual(self.actual["sampled_day_count"],150)
        self.assertEqual(self.actual["window_starts"],self.saved["window_starts"])
        starts=[date.fromisoformat(x) for x in self.actual["window_starts"]]
        self.assertTrue(all(b-a>=timedelta(days=5) for a,b in zip(starts,starts[1:])))

    def test_metrics_reproducible(self):
        for unit in ("records","positive_cells"):
            actual=self.actual[unit]
            reference=self.saved[unit]
            for metric in ("median_abs_pct_error","median_bias_pct",
                           "spearman","coverage_80pct_nominal",
                           "residual_quantile_low","residual_quantile_high"):
                self.assertAlmostEqual(actual[metric],reference[metric],places=7)
            self.assertTrue(actual["no_fill_applied"])
            self.assertEqual(actual["n_evaluated"],150)
        self.assertAlmostEqual(self.actual["records"]["median_abs_pct_error"],
                               31.5191014961,places=5)
        self.assertAlmostEqual(self.actual["positive_cells"]["median_abs_pct_error"],
                               34.3460087930,places=5)
