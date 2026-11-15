"""Offline tests for FireLens V10 regional archive processor.

Run: python -m unittest discover -s v10 -p 'test_*.py' -v
"""
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from prepare_region import prepare


class RegionPreparationTests(unittest.TestCase):
    def test_csv_zip_cross_sensor_union_and_annual_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            modis = root / "modis.csv"
            viirs_zip = root / "viirs.zip"
            pd.DataFrame([
                {"latitude": -5, "longitude": -60, "acq_date": "2024-01-01", "instrument": "MODIS", "type": 0},
                {"latitude": -5, "longitude": -60, "acq_date": "2024-01-01", "instrument": "MODIS", "type": 0},  # same cell/date
                {"latitude": -5, "longitude": -61, "acq_date": "2024-01-02", "instrument": "MODIS", "type": 0},
                {"latitude": -5, "longitude": -20, "acq_date": "2024-01-01", "instrument": "MODIS", "type": 0},  # outside AOI
            ]).to_csv(modis, index=False)
            viirs_csv = root / "viirs.csv"
            pd.DataFrame([
                {"latitude": -5, "longitude": -60, "acq_date": "2024-01-01", "instrument": "SNPP", "type": 0},
                {"latitude": -5, "longitude": -62, "acq_date": "2024-01-02", "instrument": "SNPP", "type": 0},
            ]).to_csv(viirs_csv, index=False)
            with zipfile.ZipFile(viirs_zip, "w") as z:
                z.write(viirs_csv, arcname="nested/viirs.csv")

            out = root / "result"
            inv = prepare("amazon", [modis], [viirs_zip], out,
                          start="2024-01-01", end="2024-01-03")
            days = pd.read_csv(out / "daily_summary.csv")
            jan1, jan2, jan3 = [days.iloc[i] for i in range(3)]
            self.assertEqual(int(jan1["modis_record_rows"]), 2)
            self.assertEqual(int(jan1["viirs_record_rows"]), 1)
            self.assertEqual(int(jan1["modis_positive_cells"]), 1)
            self.assertEqual(int(jan1["both_positive_cells"]), 1)
            self.assertEqual(int(jan1["union_positive_cells"]), 1)
            self.assertEqual(int(jan2["union_positive_cells"]), 2)
            self.assertEqual(int(jan2["both_positive_cells"]), 0)
            self.assertEqual(int(jan3["union_positive_cells"]), 0)
            self.assertEqual(inv["row_counts"]["MODIS"]["outside_scope_rows"], 1)
            self.assertFalse((out / "staging_cells.sqlite").exists())
            self.assertTrue((out / "grid_10km_daily.csv.gz").exists())
            spatial=pd.read_csv(out / "annual_spatial_1deg.csv")
            self.assertEqual(int(spatial["union_cell_days"].sum()),3)
            annual = pd.read_csv(out / "annual_summary.csv")
            self.assertEqual(int(annual.loc[0, "union_positive_cell_days"]), 3)

    def test_wrong_sensor_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wrong = root / "wrong.csv"
            right = root / "right.csv"
            pd.DataFrame([{"latitude": -5, "longitude": -60,
                           "acq_date": "2024-01-01", "instrument": "SNPP", "type": 0}]).to_csv(wrong, index=False)
            pd.DataFrame([{"latitude": -5, "longitude": -60,
                           "acq_date": "2024-01-01", "instrument": "SNPP", "type": 0}]).to_csv(right, index=False)
            with self.assertRaisesRegex(ValueError, "instrument"):
                prepare("amazon", [wrong], [right], root / "out",
                        start="2024-01-01", end="2024-01-03")

    def test_duplicate_archive_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sensor_file = root / "same.csv"
            pd.DataFrame([{"latitude": -5, "longitude": -60,
                           "acq_date": "2024-01-01", "type": 0}]).to_csv(sensor_file, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate input archive"):
                prepare("amazon", [sensor_file], [sensor_file], root / "out",
                        start="2024-01-01", end="2024-01-03")

if __name__ == "__main__":
    unittest.main()
