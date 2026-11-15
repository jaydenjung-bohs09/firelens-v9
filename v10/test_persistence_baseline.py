import gzip
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from persistence_baseline import compute


class PersistenceTests(unittest.TestCase):
    def test_day_by_day_proxy_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            days=pd.DataFrame([
                ("2022-01-01",2022,1,1,True,True),
                ("2022-01-02",2022,2,2,True,True),
                ("2022-01-03",2022,3,1,True,True),
            ],columns=["date_utc","year","modis_positive_cells","union_positive_cells",
                      "modis_has_positive_records","viirs_has_positive_records"])
            days.to_csv(p/"daily.csv",index=False)
            with gzip.open(p/"grid.csv.gz","wt") as handle:
                handle.write("date_utc,grid_x,grid_y,modis_positive,viirs_positive\n")
                handle.write("2022-01-01,0,0,1,1\n")
                handle.write("2022-01-02,0,0,1,1\n")
                handle.write("2022-01-02,1,0,1,1\n")
                handle.write("2022-01-03,1,0,1,1\n")
            detail,annual=compute(p/"daily.csv",p/"grid.csv.gz",p/"out")
            self.assertEqual(detail.loc[1,["tp_positive_cells","fp_positive_cells",
                                           "fn_positive_cells"]].tolist(),[1,0,1])
            self.assertEqual(detail.loc[2,["tp_positive_cells","fp_positive_cells",
                                           "fn_positive_cells"]].tolist(),[1,1,0])
            self.assertAlmostEqual(float(annual.iloc[0]["f1_recorded_positive_overlap"]),2/3,places=5)
            self.assertEqual(int(annual.iloc[0]["days"]),2)
            self.assertTrue((p/"out"/"persistence_baseline_daily.csv").exists())

if __name__=="__main__":
    unittest.main()
