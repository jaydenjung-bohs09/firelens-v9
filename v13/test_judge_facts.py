"""Judge-visible scientific facts from original provided NASA FIRMS archives."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import jensenshannon

ROOT=Path(__file__).resolve().parents[1]

class TestJudgeEvidence(unittest.TestCase):
    def test_park_viirs_outage_is_in_source_not_join(self):
        a=pd.read_csv(ROOT/"data"/"historic"/"california_july_2024_hotspots.csv")
        a=a.loc[a.latitude.between(39.2,40.3)&
                a.longitude.between(-122.5,-121.0)]
        viirs=a.loc[a.sensor_group.eq("VIIRS")].groupby("date").size()
        modis=a.loc[a.sensor_group.eq("MODIS")].groupby("date").size()
        for day in ("2024-07-24","2024-07-25","2024-07-26",
                    "2024-07-27","2024-07-28"):
            self.assertEqual(int(viirs.get(day,0)),0,day)
        self.assertEqual(int(viirs.get("2024-07-29",0)),187)
        self.assertEqual(int(modis.get("2024-07-25",0)),603)
        self.assertEqual(int(modis.get("2024-07-26",0)),607)
        status=pd.read_csv(ROOT/"data"/"historic"/"july_2024_sensor_status.csv")
        s=status.set_index("UTC date")
        for day in ("2024-07-25","2024-07-26","2024-07-27","2024-07-28"):
            self.assertTrue(bool(s.loc[day,"FIRMS listed VIIRS missing date"]))
            self.assertTrue(bool(s.loc[day,"S-NPP anomaly window"]))

    def test_screen_excludes_known_park_outage(self):
        from critical_periods import seasonal_screen
        manifest=pd.read_csv(ROOT/"data"/"predict"/"sequence_manifest.csv")
        daily,periods=seasonal_screen(manifest)
        d=daily.set_index(daily.date.dt.strftime("%Y-%m-%d"))
        for day in pd.date_range("2024-07-24","2024-07-28"):
            row=d.loc[day.strftime("%Y-%m-%d")]
            self.assertFalse(bool(row["comparison_eligible"]))
            self.assertFalse(bool(row["candidate_day"]))

    def test_2024_separate_data_gap_review_track(self):
        manifest=pd.read_csv(ROOT/"data"/"predict"/"sequence_manifest.csv")
        subset=manifest.loc[manifest.target_date_utc.str.startswith("2024")]
        gap=(subset.target_snpp_known_anomaly_window.astype(bool)
             &subset.target_modis_has_any_hotspot_record.astype(bool)
             &~subset.target_viirs_has_any_hotspot_record.astype(bool))
        dates=set(subset.loc[gap,"target_date_utc"])
        self.assertEqual(len(dates),11)
        for day in pd.date_range("2024-07-24","2024-07-28"):
            self.assertIn(day.strftime("%Y-%m-%d"),dates)

    def test_usa_australia_calendar_score_source(self):
        df=pd.read_csv(ROOT/"v11"/"processed"/"global"/"global_country_year.csv")
        subset=df.loc[df.sensor.eq("MODIS")&
                      df.year.between(2001,2025)&df.iso3.isin(["USA","AUS"])]
        cols=[f"m{i:02d}" for i in range(1,13)]
        values={}
        for name in ("USA","AUS"):
            v=subset.loc[subset.iso3.eq(name),cols].sum().to_numpy(float)
            values[name]=v/v.sum()
        sim=1-float(jensenshannon(values["USA"],values["AUS"],base=2))
        self.assertAlmostEqual(sim,.70611608335,places=8)
        self.assertEqual(int(np.argmax(values["USA"])),7)  # August
        self.assertEqual(int(np.argmax(values["AUS"])),9)  # October

    def test_model_grid_and_contextual_california_outline(self):
        card=json.loads((ROOT/"data"/"predict"/"dataset_card.json").read_text())
        geom=json.loads((ROOT/"v13"/"california_outline.geojson").read_text())
        self.assertEqual(geom["properties"]["name"],"California")
        bbox=card["fixed_bbox_lonlat"]
        t=Transformer.from_crs("EPSG:4326","EPSG:6933",always_xy=True)
        x0,y0=t.transform(bbox[0],bbox[1])
        x1,y1=t.transform(bbox[2],bbox[3])
        size=10000
        xmin=math.floor(min(x0,x1)/size)
        xmax=math.floor(max(x0,x1)/size)
        ymin=math.floor(min(y0,y1)/size)
        ymax=math.floor(max(y0,y1)/size)
        self.assertEqual((ymax-ymin+1,xmax-xmin+1),(99,103))
        verts=np.array(geom["geometry"]["coordinates"][0])
        x,y=t.transform(verts[:,0],verts[:,1])
        px=np.asarray(x)/size-xmin-.5
        py=np.asarray(y)/size-ymin-.5
        self.assertGreater(float(((px>=-.5)&(px<=103)&
                                  (py>=-.5)&(py<=99)).mean()),.95)

if __name__=="__main__":
    unittest.main()
