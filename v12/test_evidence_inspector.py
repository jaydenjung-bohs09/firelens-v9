"""Regression tests for time-aware Park Fire Evidence Inspector.

Uses unchanged V8 historical export and positive-record audit; no fabricated
negative observations or satellite opportunity assumptions.
"""
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v12.evidence_inspector import (
    EVIDENCE_ORDER, match_acquisition_times, prepare_park_records,
    rebuild_evidence_groups
)
HIST = ROOT / "data" / "historic"


class TestEvidenceInspector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pd.read_csv(HIST / "park_hotspot_records.csv")
        cls.status = pd.read_csv(HIST / "july_2024_sensor_status.csv")
        cls.legacy = pd.read_csv(HIST / "park_evidence_groups.csv")
        cls.baseline = rebuild_evidence_groups(
            cls.records, cls.status, tolerance_minutes=180,
            type0_only=False
        )

    def test_default_totals_reproduce_legacy_v8(self):
        g=self.baseline
        self.assertEqual(len(g),316)
        self.assertEqual(int(g["Both sensors positive"].sum()),54)
        self.assertEqual(int(g["Time-compatible"].sum()),45)
        self.assertEqual(int(g["One-to-one matched pairs"].sum()),149)
        self.assertFalse(g["Observation coverage verified"].any())

    def test_all_groups_and_pair_counts_match_frozen_reference(self):
        keys=["UTC date","Cell ID","Day/night"]
        actual=self.baseline.set_index(keys).sort_index()
        legacy=self.legacy.set_index(keys).sort_index()
        self.assertEqual(set(actual.index),set(legacy.index))
        for new,old in [
            ("MODIS records","MODIS records"),
            ("VIIRS records","VIIRS records"),
            ("Time-compatible","At least one time-matched pair"),
            ("One-to-one matched pairs","Matched pairs")
        ]:
            pd.testing.assert_series_equal(
                actual[new].rename("x"), legacy[old].rename("x"),
                check_names=False
            )
        pairs=actual["Closest acquisition gap (min)"]
        both=actual["Both sensors positive"]
        self.assertTrue(pairs.loc[both].notna().all())
        self.assertTrue(pairs.loc[~both].isna().all())

    def test_changing_window_is_monotonic(self):
        zero=rebuild_evidence_groups(self.records,self.status,0)
        short=rebuild_evidence_groups(self.records,self.status,60)
        long=rebuild_evidence_groups(self.records,self.status,360)
        a=[int(x["Time-compatible"].sum()) for x in [zero,short,self.baseline,long]]
        self.assertEqual(a,sorted(a))
        self.assertTrue(a[-1] >= 45)

    def test_one_to_one_matching_and_greedy_order(self):
        self.assertEqual(
            match_acquisition_times([100,100,500],[102,110,505],10),
            [(100.,102.,2.),(100.,110.,10.),(500.,505.,5.)]
        )
        self.assertEqual(match_acquisition_times([10],[200],20),[])
        with self.assertRaises(ValueError):
            match_acquisition_times([0],[0],-1)

    def test_known_missing_data_date_is_not_coverage_claim(self):
        g=self.baseline
        july25=g.loc[g["UTC date"].eq("2024-07-25")]
        self.assertFalse(july25.empty)
        self.assertTrue(july25["S-NPP anomaly window"].all())
        self.assertTrue(july25["FIRMS listed VIIRS missing date"].all())
        modis_only=july25.loc[july25["MODIS records"].gt(0) &
                              july25["VIIRS records"].eq(0)]
        self.assertTrue(modis_only["Evidence class"].eq(EVIDENCE_ORDER[2]).all())
        self.assertFalse(july25["Observation coverage verified"].any())

    def test_type0_filter_changes_underlying_record_counts(self):
        prepared=prepare_park_records(self.records)
        self.assertEqual(len(prepared),3092)
        self.assertEqual(int(prepared["type"].eq(2).sum()),8)
        filtered=rebuild_evidence_groups(self.records,self.status,180,True)
        self.assertLessEqual(len(filtered),len(self.baseline))
        self.assertLessEqual(int(filtered["MODIS records"].sum() +
                                 filtered["VIIRS records"].sum()),3084)

if __name__=="__main__":
    unittest.main()
