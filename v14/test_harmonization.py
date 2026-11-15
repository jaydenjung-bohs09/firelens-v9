"""Unit tests for common active-cell-day harmonization unit."""
import unittest
import pandas as pd
from v14.harmonization import active_cell_days


class TestCommonCellDay(unittest.TestCase):
    @staticmethod
    def sample():
        return pd.DataFrame([
            ("MODIS","2024-07-24",39.4,-121.9,0,"D"),
            ("MODIS","2024-07-24",39.4,-121.9,0,"D"), # same active cell
            ("VIIRS","2024-07-24",39.4,-121.9,0,"N"),
            ("VIIRS","2024-07-24",39.4,-121.9,2,"D"), # nonvegetation
            ("MODIS","2024-07-25",39.4,-121.9,0,"D"),
            ("VIIRS","2024-07-25",39.4,-121.9,0,"D"),
        ],columns=["sensor_group","acq_date","latitude","longitude","type","daynight"])

    def test_union_positive_default(self):
        table=active_cell_days(self.sample(),outage_dates=["2024-07-24"])
        self.assertEqual(table.columns.tolist(),[
            "cell_id","date","modis_active","viirs_active","outage_flag"])
        self.assertEqual(len(table),2)
        self.assertEqual(
            table.set_index("date")[["modis_active","viirs_active","outage_flag"]].to_dict("index"),
            {"2024-07-24":{"modis_active":1,"viirs_active":1,"outage_flag":1},
             "2024-07-25":{"modis_active":1,"viirs_active":1,"outage_flag":0}}
        )
        self.assertTrue(table.cell_id.str.count("_").eq(1).all())

    def test_daynight_split(self):
        rows=active_cell_days(self.sample(),separate_daynight=True,
                              outage_dates=["2024-07-24"])
        self.assertEqual(len(rows),3)
        day=rows.loc[rows.date.eq("2024-07-24") & rows.daynight.eq("D")].iloc[0]
        night=rows.loc[rows.date.eq("2024-07-24") & rows.daynight.eq("N")].iloc[0]
        self.assertEqual((day.modis_active,day.viirs_active),(1,0))
        self.assertEqual((night.modis_active,night.viirs_active),(0,1))
        self.assertEqual((day.outage_flag,night.outage_flag),(1,1))

    def test_type0_filter_avoids_false_zeros(self):
        bad=self.sample().loc[lambda x:x.type.eq(2)]
        self.assertTrue(active_cell_days(bad).empty)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            active_cell_days(self.sample(),grid_m=0)
        with self.assertRaises(ValueError):
            active_cell_days(self.sample().drop(columns="longitude"))
        with self.assertRaises(ValueError):
            active_cell_days(self.sample().assign(sensor_group="UNKNOWN"))
