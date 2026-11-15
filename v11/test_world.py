"""Automated NASA global archive audit without needing the 4.6-GB ZIPs in CI."""
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"processed"/"global"
MONTHS=[f"m{i:02d}" for i in range(1,13)]


class GlobalAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=pd.read_csv(DATA/"global_country_year.csv",dtype={"iso3":"string"})
        cls.manifest=json.loads((DATA/"global_manifest.json").read_text())

    def test_source_inventory_totals(self):
        self.assertEqual(len(self.df),7033)
        self.assertEqual(len(self.df),self.manifest["country_year_rows"])
        self.assertEqual(int(self.df.vegetation_records.sum()),252915305)
        self.assertEqual(int(self.df.vegetation_records.sum()),
                         self.manifest["total_vegetation_records"])
        self.assertEqual(int(self.df.raw_records.sum()),267356689)
        self.assertEqual(int(self.df.raw_records.sum()),
                         self.manifest["total_raw_records"])
        self.assertFalse(self.df.iso3.isna().any())
        self.assertFalse(self.df.duplicated(["sensor","year","iso3"]).any())

    def test_calendar_totals(self):
        totals=self.df[MONTHS].sum(axis=1).to_numpy()
        self.assertTrue(np.array_equal(totals,
                         self.df.vegetation_records.to_numpy()))
        self.assertTrue((self.df.vegetation_records>=0).all())

    def test_archive_checksums_and_sensor_totals(self):
        archive=pd.read_csv(DATA/"archive_checksums.csv")
        self.assertEqual(len(archive),34)
        self.assertFalse(archive.archive.duplicated().any())
        self.assertTrue(archive.sha256.astype(str).str.fullmatch(r"[a-f0-9]{64}").all())
        self.assertEqual(int(archive.raw_rows.sum()),267356689)
        self.assertEqual(int(archive.type0_rows.sum()),252915305)
        for row in archive.itertuples(index=False):
            product="MODIS" if row.archive.startswith("modis_") else "VIIRS_NOAA20"
            year=int(row.archive.split("_")[1])
            subset=self.df.loc[self.df.sensor.eq(product)&self.df.year.eq(year)]
            self.assertEqual(int(subset.raw_records.sum()),row.raw_rows)
            self.assertEqual(int(subset.vegetation_records.sum()),row.type0_rows)

    def test_sensor_periods(self):
        years=self.df.groupby("sensor").year.agg(["min","max"]).to_dict("index")
        self.assertEqual(years["MODIS"],{"min":2000,"max":2025})
        self.assertEqual(years["VIIRS_NOAA20"],{"min":2018,"max":2025})
        self.assertEqual(self.df[self.df.sensor.eq("MODIS")].year.nunique(),26)
        self.assertEqual(self.df[self.df.sensor.eq("VIIRS_NOAA20")].year.nunique(),8)

    def test_country_reconciliation(self):
        for code in ["USA","BRA","AUS","COD","RUS"]:
            self.assertGreater(
                self.df.loc[(self.df.iso3.eq(code)) &
                            (self.df.year.eq(2024)) &
                            (self.df.sensor.eq("MODIS")),
                            "vegetation_records"].sum(),0)
        self.assertEqual(int(
            self.df[(self.df.iso3=="BRA") & (self.df.year==2024) &
                    (self.df.sensor=="MODIS")].vegetation_records.sum()
        ),521413)

if __name__=="__main__":
    unittest.main()
