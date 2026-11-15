"""Tier-1 cross-validation regression checks against committed source-derived CSV."""
import unittest
import numpy as np
import pandas as pd
from v15.calibration import country_month_rows,complete_overlap_years
from v15.validation import (
    VALIDATION_CSV,validation_predictions,summarize_predictions,selected_method,
)

class HeldOutValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=country_month_rows()
        cls.years=complete_overlap_years(cls.rows)
        cls.pred=validation_predictions(cls.rows)
        cls.report=summarize_predictions(cls.pred)
        cls.saved=pd.read_csv(VALIDATION_CSV)

    def test_2018_partial_year_excluded_and_every_year_withheld(self):
        self.assertEqual(self.years,tuple(range(2019,2026)))
        self.assertEqual(set(self.pred.year.unique()),set(self.years))
        self.assertEqual(set(self.pred.method),{"none","ratio","log","global_ratio"})

    def test_recomputed_values_match_saved_csv(self):
        keys=["level","continent","method"]
        actual=self.report.set_index(keys).sort_index()
        saved=self.saved.set_index(keys).sort_index()
        self.assertEqual(actual.index.tolist(),saved.index.tolist())
        for col in ("n_source_pairs","n_predictions","n_pct_valid","n_modis_zero"):
            np.testing.assert_array_equal(actual[col],saved[col])
        for col in ("median_abs_pct_error","median_bias_pct","spearman"):
            np.testing.assert_allclose(actual[col],saved[col],rtol=0,atol=5.1e-8)

    def test_winner_uses_held_out_mape_not_correlation(self):
        self.assertEqual(selected_method(self.report),"ratio")
        g=self.report.loc[self.report.level.eq("global")].set_index("method")
        self.assertAlmostEqual(g.loc["none","median_abs_pct_error"],80.65116279,places=6)
        self.assertAlmostEqual(g.loc["ratio","median_abs_pct_error"],20.42606516,places=6)
        self.assertAlmostEqual(g.loc["log","median_abs_pct_error"],21.46132535,places=6)

    def test_unobserved_modis_zero_is_not_prediction(self):
        for method,part in self.pred.groupby("method"):
            self.assertTrue(part.loc[part.modis.eq(0),"predicted"].isna().all())
