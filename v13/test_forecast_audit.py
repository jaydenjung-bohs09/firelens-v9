"""Reproduce archived model baseline and block-bootstrap evidence."""
import json
import unittest
from pathlib import Path
from v13.forecast_audit import compute, ROOT

class TestForecastAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=compute(bootstrap_replicates=3000)
        cls.expected=json.loads((ROOT/"v13"/"forecast_audit_results.json").read_text())
    def test_common_dates(self):
        self.assertEqual(self.report["common_dates"],285)
        self.assertEqual(self.report["first_date"],"2025-03-22")
        self.assertEqual(self.report["last_date"],"2025-12-31")
        self.assertEqual(self.report["locked_model_threshold"],.675)
    def test_matches_preserved_results(self):
        for name,expected in self.expected["methods"].items():
            actual=self.report["methods"][name]
            for column in ("tp","fp","fn"):
                self.assertEqual(actual[column],expected[column],(name,column))
            self.assertAlmostEqual(actual["f1"],expected["f1"],places=9)
            for a,b in zip(actual["ci95_f1"],expected["ci95_f1"]):
                self.assertAlmostEqual(a,b,places=8)
            if name!="ConvLSTM":
                self.assertAlmostEqual(actual["model_minus_baseline_f1"],
                                       expected["model_minus_baseline_f1"],places=9)
                for a,b in zip(actual["ci95_model_minus_baseline"],
                               expected["ci95_model_minus_baseline"]):
                    self.assertAlmostEqual(a,b,places=8)
    def test_baselines_not_test_tuned(self):
        self.assertGreater(self.report["methods"]["ConvLSTM"]["f1"],
                           self.report["methods"]["At least three of seven days"]["f1"])
        self.assertGreater(self.report["methods"]["ConvLSTM"]["f1"],
                           self.report["methods"]["One-day persistence"]["f1"])
        self.assertGreater(self.report["methods"]["ConvLSTM"]["f1"],
                           self.report["methods"]["Seven-day union"]["f1"])
        self.assertGreater(self.report["methods"]["One-day persistence"]["model_minus_baseline_f1"],0)
if __name__=="__main__":
    unittest.main()
