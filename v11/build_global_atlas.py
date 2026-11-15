#!/usr/bin/env python3
"""Rebuild FireLens V11 global country/year table from original NASA FIRMS ZIPs.

Input: modis_2000_all_countries.zip ... modis_2025_all_countries.zip and
viirs-jpss1_2018_all_countries.zip ... viirs-jpss1_2025_all_countries.zip.
Output: world-positive-record summary CSV + provenance JSON with SHA256 checksums.

CSV members are streamed, one country at a time. Full raw archives are not
committed to GitHub. NASA type=0 is presumed vegetation fire, NOT a ground-truth
label or an independent wildfire count.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
import pycountry

MATCH=re.compile(r"^(modis|viirs-jpss1)_(\d{4})_all_countries\.zip$")
ALIAS={
 "Bolivia":"BOL","Brunei":"BRN","Cote d Ivoire":"CIV",
 "Democratic Republic of the Congo":"COD","Republic of the Congo":"COG",
 "Republic of Congo":"COG","Russian Federation":"RUS",
 "United States":"USA","Vietnam":"VNM","Tanzania":"TZA",
 "Syria":"SYR","Iran":"IRN","Turkey":"TUR","Kosovo":"XKX",
 "Laos":"LAO","Lao PDR":"LAO","Venezuela":"VEN",
 "North Korea":"PRK","Dem Rep Korea":"PRK","Republic of Korea":"KOR",
 "South Korea":"KOR","Moldova":"MDA","Palestine":"PSE",
 "Micronesia":"FSM","Taiwan":"TWN","United States Virgin Islands":"VIR",
 "Czech Republic":"CZE","Aland Islands":"ALA","Cape Verde":"CPV",
 "Curacao":"CUW","Reunion":"REU","Timor Leste":"TLS",
 "Sao Tome and Principe":"STP","Antarctica":"ATA","Eswatini":"SWZ",
 "Swaziland":"SWZ","Faeroe Islands":"FRO","Falkland Islands":"FLK",
 "French Southern and Antarctic Lands":"ATF",
 "Heard I and McDonald Islands":"HMD",
 "Macedonia Former Yugoslav Republic of":"MKD",
 "Saint Helena":"SHN","Saint-Barthelemy":"BLM","Saint-Martin":"MAF",
 "Sint Maarten":"SXM","South Georgia and South Sandwich Islands":"SGS",
 "The Gambia":"GMB","Wallis and Futuna Islands":"WLF"
}
MCOLS=[f"m{i:02d}" for i in range(1,13)]
FIELDS=["sensor","year","country","iso3","raw_records","vegetation_records",
        "nonvegetation_records","invalid_rows","excluded_out_of_year",
        "active_record_days","day_records","night_records"]+MCOLS

def to_iso(name):
    if name in ALIAS: return ALIAS[name]
    try: return pycountry.countries.lookup(name).alpha_3
    except LookupError: return ""

def process_archive(path):
    path=Path(path)
    m=MATCH.match(path.name)
    if m is None: raise ValueError("Unknown archive "+path.name)
    product,ys=m.groups();year=int(ys)
    sensor="MODIS" if product=="modis" else "VIIRS_NOAA20"
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(4*1024*1024),b""):
            digest.update(block)
    country_rows=[]
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".csv"): continue
            stem=Path(info.filename).stem
            prefix=f"{product}_{year}_"
            if not stem.startswith(prefix):
                raise ValueError("Unexpected member "+info.filename)
            country=stem[len(prefix):].replace("_"," ")
            months=[0]*12;days=set();raw=veg=other=invalid=outside=day=night=0
            with archive.open(info) as stream:
                header=stream.readline().strip().lower().split(b",")
                if header.index(b"acq_date")!=5 or header.index(b"type")!=len(header)-1 or header.index(b"daynight")!=len(header)-2:
                    raise ValueError("Unexpected column layout "+info.filename)
                for line in stream:
                    if not line.strip(): continue
                    raw+=1
                    try: beginning,dn,kind=line.rstrip(b"\r\n").rsplit(b",",2)
                    except ValueError:
                        invalid+=1;continue
                    if kind!=b"0":
                        other+=1;continue
                    fields=beginning.split(b",",6)
                    if len(fields)<6: invalid+=1;continue
                    utc_date=fields[5]
                    try:
                        observed_year=int(utc_date[:4]);month=int(utc_date[5:7])
                    except ValueError:
                        invalid+=1;continue
                    if observed_year!=year or not 1<=month<=12:
                        outside+=1;continue
                    veg+=1;months[month-1]+=1;days.add(utc_date)
                    if dn==b"D": day+=1
                    if dn==b"N": night+=1
            if veg!=sum(months):
                raise AssertionError("Month/record mismatch "+country)
            country_rows.append(dict(
                sensor=sensor,year=year,country=country,iso3=to_iso(country),
                raw_records=raw,vegetation_records=veg,
                nonvegetation_records=other,invalid_rows=invalid,
                excluded_out_of_year=outside,
                active_record_days=len(days),day_records=day,night_records=night,
                **{MCOLS[i]:months[i] for i in range(12)}
            ))
    return {"filename":path.name,"sensor":sensor,"year":year,
            "sha256":digest.hexdigest(),"rows":country_rows}

def rebuild(directory:Path,out:Path,workers:int=6):
    paths=sorted(x for x in directory.glob("*.zip") if MATCH.match(x.name))
    expected={f"modis_{y}_all_countries.zip" for y in range(2000,2026)}
    expected|={f"viirs-jpss1_{y}_all_countries.zip" for y in range(2018,2026)}
    if {x.name for x in paths}!=expected:
        raise ValueError("Input inventory must contain exactly 26 MODIS + 8 NOAA-20 archives")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        products=list(pool.map(process_archive,map(str,paths)))
    rows=[r for a in products for r in a["rows"]]
    if len(rows)!=7033:
        raise AssertionError(f"Expected 7,033 country/year rows, got {len(rows)}")
    if sum(r["vegetation_records"] for r in rows)!=252915305:
        raise AssertionError("Aggregated historical total differs from audited source")
    if any(not r["iso3"] for r in rows):
        raise ValueError("A country/territory requires an ISO-3 alias")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"global_country_year.csv").open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    provenance={
        "archive_count":len(products),
        "country_year_rows":len(rows),
        "total_raw_records":sum(r["raw_records"] for r in rows),
        "total_vegetation_records":sum(r["vegetation_records"] for r in rows),
        "type_filter":"Only NASA FIRMS type=0 presumed vegetation fire",
        "interpretation":"Not a unique wildfire inventory or harmonized satellite rate",
        "archives":[
            {"filename":a["filename"],"sensor":a["sensor"],"year":a["year"],
             "sha256":a["sha256"],
             "raw_rows":sum(r["raw_records"] for r in a["rows"]),
             "type0_rows":sum(r["vegetation_records"] for r in a["rows"])}
            for a in products
        ]
    }
    (out/"full_source_manifest.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=6)
    args=ap.parse_args()
    result=rebuild(args.input,args.output,args.workers)
    print(json.dumps({k:v for k,v in result.items() if k!="archives"},indent=2))
