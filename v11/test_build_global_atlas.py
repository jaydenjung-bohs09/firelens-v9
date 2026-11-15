"""Validate streaming NASA FIRMS member parsing against synthetic CSV archives."""
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from build_global_atlas import process_archive, to_iso


class OriginalArchiveTests(unittest.TestCase):
    def test_modis_csv_type_filter_and_months(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"modis_2024_all_countries.zip"
            header=("latitude,longitude,brightness,scan,track,acq_date,"
                    "acq_time,satellite,instrument,confidence,version,"
                    "bright_t31,frp,daynight,type\n")
            entries=[
                "-5,-60,330,1,1,2024-01-14,1340,Terra,MODIS,80,6.1,300,9,D,0\n",
                "-5,-60,330,1,1,2024-08-14,1340,Terra,MODIS,80,6.1,300,9,N,0\n",
                "-5,-60,330,1,1,2024-08-14,1340,Terra,MODIS,80,6.1,300,9,N,2\n"
            ]
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("modis/2024/modis_2024_Brazil.csv",header+"".join(entries))
            result=process_archive(str(path))
            self.assertEqual(result["year"],2024)
            self.assertEqual(result["sensor"],"MODIS")
            self.assertEqual(result["rows"][0]["country"],"Brazil")
            self.assertEqual(result["rows"][0]["iso3"],"BRA")
            self.assertEqual(result["rows"][0]["raw_records"],3)
            self.assertEqual(result["rows"][0]["vegetation_records"],2)
            self.assertEqual(result["rows"][0]["nonvegetation_records"],1)
            self.assertEqual(result["rows"][0]["m01"],1)
            self.assertEqual(result["rows"][0]["m08"],1)
            self.assertEqual(result["rows"][0]["active_record_days"],2)
            self.assertEqual(result["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_country_aliases(self):
        self.assertEqual(to_iso("Democratic Republic of the Congo"),"COD")
        self.assertEqual(to_iso("Republic of Congo"),"COG")
        self.assertEqual(to_iso("Dem Rep Korea"),"PRK")
        self.assertEqual(to_iso("Russian Federation"),"RUS")


if __name__=="__main__":
    unittest.main()
