"""Scientific regression tests for archived country-month calibration."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from v15.calibration import (
    ROOT,REGIONS,MONTH_COLS,country_month_rows,complete_overlap_years,
    fit_calibration,available_pairs
)

class CalibrationTests(unittest.TestCase):
    def synthetic(self):
        rng=np.random.default_rng(123)
        records=[]
        for year in range(2019,2025):
            for month in range(1,13):
                m=int(rng.integers(300,20000))
                records.append(dict(iso3="AUS",country="Australia",
                                    continent="Oceania",year=year,month=month,
                                    modis=m,viirs=2*m))
        return pd.DataFrame(records)

    def test_two_times_modis_recovers_log_slope(self):
        rows=self.synthetic()
        fit=fit_calibration(rows,years=tuple(range(2019,2025)))
        self.assertAlmostEqual(fit.beta_global[1],1,delta=.08)
        self.assertAlmostEqual(fit.beta_global[0],np.log(2),delta=.28)
        for method in ("ratio","log"):
            predicted=fit.predict(rows,method)
            self.assertAlmostEqual(float((predicted.predicted/rows.viirs).median()),
                                   1,delta=.08)

    def test_single_global_ratio_equals_country_month_ratio_when_uniform(self):
        rows=self.synthetic()
        additional=rows.copy()
        additional["iso3"]="BRA"
        additional["country"]="Brazil"
        additional["continent"]="South America"
        additional["modis"]=additional["modis"]*3
        additional["viirs"]=additional["modis"]*2
        same_ratio=pd.concat([rows,additional],ignore_index=True)
        fit=fit_calibration(same_ratio,years=tuple(range(2019,2025)))
        global_pred=fit.predict(same_ratio,"global_ratio").predicted.to_numpy()
        country_pred=fit.predict(same_ratio,"ratio").predicted.to_numpy()
        np.testing.assert_allclose(global_pred,country_pred,rtol=1e-12,atol=1e-9)

    def test_zero_modis_returns_nan_and_conditional_viirs_probability(self):
        rows=self.synthetic()
        extra=dict(iso3="AUS",country="Australia",continent="Oceania",
                   year=2024,month=12,modis=0,viirs=5)
        rows=pd.concat([rows,pd.DataFrame([extra])],ignore_index=True)
        fit=fit_calibration(rows,years=tuple(range(2019,2025)))
        for method in ("none","ratio","log","global_ratio"):
            result=fit.predict(rows.iloc[[-1]],method)
            self.assertTrue(np.isnan(result.predicted.iloc[0]))
            self.assertTrue(0<result.p_viirs_positive_when_modis_zero.iloc[0]<1)

    def test_source_years_and_monthly_audit(self):
        rows=country_month_rows()
        self.assertEqual(complete_overlap_years(rows),tuple(range(2019,2026)))
        self.assertGreater(len(available_pairs(rows)),15000)
        self.assertTrue(rows.loc[rows.year.eq(2018)&rows.month.eq(1),"viirs"].notna().any())

    def test_missing_columns_raise(self):
        data=pd.DataFrame([dict(sensor="MODIS",year=2019,country="USA",
                               iso3="USA",vegetation_records=10)])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.csv"
            data.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"missing columns"):
                country_month_rows(path)
