"""Reproduce complete-year Amazon record vs positive-cell validation."""
import json
import unittest
import numpy as np
import pandas as pd
from v15.amazon_tier2 import ROOT,DATA,STATUS,complete_years,validate,fit,predict,UNITS

class AmazonTier2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(DATA)
        cls.saved=json.loads((ROOT/"v15/amazon_tier2_results.json").read_text())
        cls.current=validate()

    def test_last_2025_year_complete(self):
        self.assertEqual(complete_years(self.data),(2022,2023,2024,2025))
        self.assertEqual(self.current["holdout_year"],2025)
        self.assertEqual(self.current["known_s_npp_flagged_dates_in_training"],14)

    def test_results_reproduced(self):
        for unit in UNITS:
            a=self.current["units"][unit]
            b=self.saved["units"][unit]
            self.assertEqual(a["winning_method"],b["winning_method"])
            self.assertEqual(a["fitted_days"],1081)
            for method in ("none","ratio","log"):
                for metric in ("median_abs_pct_error","median_bias_pct","spearman"):
                    self.assertAlmostEqual(
                        a["methods"][method][metric],
                        b["methods"][method][metric],places=7)

    def test_raw_record_units_lower_held_out_error(self):
        a=self.current["units"]["records"]["methods"]["ratio"]["median_abs_pct_error"]
        b=self.current["units"]["positive_cells"]["methods"]["ratio"]["median_abs_pct_error"]
        self.assertLess(a,b)
        self.assertAlmostEqual(a,28.1157996567,places=5)
        self.assertAlmostEqual(b,30.8841906579,places=5)

    def test_modis_zero_does_not_imply_viirs_zero(self):
        train=self.data.loc[self.data.year.lt(2025)]
        fitted=fit(train,"records",set())
        held=self.data.loc[self.data.year.eq(2025)].iloc[[0]].copy()
        held["modis_record_rows"]=0
        for method in ("none","ratio","log"):
            self.assertTrue(np.isnan(predict(held,"records",fitted,method)[0]))
