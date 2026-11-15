"""Four geographically distinct country calendar source-integrity tests."""
import unittest
import numpy as np
import pandas as pd
from v15.calibration import country_month_rows
from v15.validation import VALIDATION_CSV,validation_predictions
from v15.calendar import (
    country_calendar,monthly_anomalies,selected_month_summary,OUTPUT_COLUMNS,
)

class CalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=country_month_rows()
        cls.summary=pd.read_csv(VALIDATION_CSV)
        cls.predictions=validation_predictions(cls.rows)

    def test_brazil_australia_usa_indonesia(self):
        for iso in ("BRA","AUS","USA","IDN"):
            with self.subTest(iso=iso):
                result=country_calendar(
                    self.rows,self.summary,self.predictions,iso)
                self.assertEqual(result.columns.tolist(),OUTPUT_COLUMNS)
                self.assertEqual(len(result),312)
                self.assertTrue(result.loc[result.year.lt(2018),"source"].eq("modeled").all())
                self.assertTrue(result.loc[
                    result.year.gt(2018)|(result.year.eq(2018)&result.month.ge(4)),
                    "source"].eq("observed").all())
                self.assertTrue(result.loc[
                    result.year.eq(2018)&result.month.le(3),"source"].eq("modeled").all())
                self.assertTrue(result.loc[
                    result.year.eq(2000)&result.month.le(10),"value"].isna().all())
                prior=result.loc[result.year.between(2001,2017)&result.value.notna()]
                self.assertGreater(len(prior),150)
                self.assertGreater((prior.lower<=prior.value).mean(),.9)
                self.assertGreater((prior.upper>=prior.value).mean(),.9)
                summary=selected_month_summary(result,2024,7)
                self.assertGreaterEqual(summary["rank"],1)
                self.assertGreaterEqual(summary["comparable_years"],20)

    def test_modis_zero_is_missing_not_observed_zero(self):
        source=self.rows.copy()
        mask=source.iso3.eq("BRA")&source.year.eq(2001)&source.month.eq(1)
        source.loc[mask,"modis"]=0
        result=country_calendar(source,self.summary,self.predictions,"BRA")
        cell=result.loc[result.year.eq(2001)&result.month.eq(1),"value"].iloc[0]
        self.assertTrue(np.isnan(cell))
