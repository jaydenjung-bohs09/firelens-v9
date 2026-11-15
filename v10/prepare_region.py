#!/usr/bin/env python3
"""Prepare NASA FIRMS MODIS + VIIRS S-NPP archive files for FireLens V10.

Inputs: historical NASA FIRMS CSV or ZIP downloads, supplied separately by sensor.
Outputs: reproducible UTC daily, yearly, and 10-km equal-area positive-record summaries.
No absence/coverage inference and no sensor-accuracy metric are made.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

REGIONS = {
    "amazon": (-74.0, -15.0, -48.0, 2.0),
    "southeast_australia": (140.0, -39.0, 154.0, -27.0),
    "central_africa": (12.0, -11.0, 34.0, 6.0),
    "siberia": (90.0, 50.0, 135.0, 70.0),
}
SENSORS = {"modis": "MODIS", "viirs": "VIIRS_S_NPP"}
REQUIRED_COLUMNS = {"latitude", "longitude", "acq_date", "type"}
CHUNK_SIZE = 100_000
GRID_METERS = 10_000
TRANSFORM = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chunks(path: Path):
    """Yield (inner CSV name, pandas chunk) without extracting arbitrary ZIP files."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [m for m in archive.namelist()
                     if m.lower().endswith(".csv") and not m.startswith("__MACOSX/")]
            if not names:
                raise ValueError(f"No CSV inside {path}")
            for name in sorted(names):
                with archive.open(name) as stream:
                    for chunk in pd.read_csv(stream, chunksize=CHUNK_SIZE, low_memory=False):
                        yield name, chunk
    elif path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            yield path.name, chunk
    else:
        raise ValueError(f"Expected .csv or .zip: {path}")


def clean_chunk(chunk: pd.DataFrame, sensor: str, bounds, start: pd.Timestamp,
                end: pd.Timestamp, counts: Counter) -> pd.DataFrame:
    chunk.columns = chunk.columns.astype(str).str.strip().str.lower()
    missing = REQUIRED_COLUMNS.difference(chunk.columns)
    if missing:
        raise ValueError(f"{sensor} file missing FIRMS columns: {sorted(missing)}")
    if "instrument" in chunk:
        values = chunk["instrument"].dropna().astype(str).str.upper().str.strip().unique()
        allowed = {"MODIS"} if sensor == "MODIS" else {"SNPP", "VIIRS"}
        unexpected = [v for v in values if v and v not in allowed]
        if unexpected:
            raise ValueError(f"File labeled {sensor}, but instrument={unexpected}")

    counts[(sensor, "input_rows")] += len(chunk)
    # Type 0 is NASA's presumed vegetation-fire classification. Preserve
    # counts for non-vegetation anomalies, but exclude those from this view.
    fire_type = pd.to_numeric(chunk["type"], errors="coerce")
    not_vegetation = fire_type.ne(0)
    counts[(sensor, "excluded_nonvegetation_or_unknown_type")] += int(not_vegetation.sum())
    chunk = chunk.loc[~not_vegetation].copy()
    lat = pd.to_numeric(chunk["latitude"], errors="coerce")
    lon = pd.to_numeric(chunk["longitude"], errors="coerce")
    dates = pd.to_datetime(chunk["acq_date"], errors="coerce", utc=True)
    invalid = (lat.isna() | lon.isna() | dates.isna()
               | ~lat.between(-90, 90) | ~lon.between(-180, 180))
    counts[(sensor, "invalid_rows")] += int(invalid.sum())
    if invalid.any():
        chunk = chunk.loc[~invalid].copy()
        lat = lat.loc[~invalid]
        lon = lon.loc[~invalid]
        dates = dates.loc[~invalid]

    west, south, east, north = bounds
    in_scope = (lon.between(west, east) & lat.between(south, north)
                & dates.between(start, end, inclusive="both"))
    counts[(sensor, "outside_scope_rows")] += int((~in_scope).sum())
    lat, lon, dates = lat.loc[in_scope], lon.loc[in_scope], dates.loc[in_scope]
    counts[(sensor, "accepted_rows")] += len(dates)
    if dates.empty:
        return pd.DataFrame(columns=["date_utc", "grid_x", "grid_y", "sensor"])

    xx, yy = TRANSFORM.transform(lon.to_numpy(), lat.to_numpy())
    return pd.DataFrame({
        "date_utc": dates.dt.strftime("%Y-%m-%d").to_numpy(),
        "grid_x": np.floor(np.asarray(xx) / GRID_METERS).astype("int64"),
        "grid_y": np.floor(np.asarray(yy) / GRID_METERS).astype("int64"),
        "sensor": sensor,
    })


def prepare(region: str, modis_files: list[Path], viirs_files: list[Path],
            output: Path, start: str = "2022-01-01", end: str = "2025-12-31",
            overwrite: bool = False, keep_sqlite: bool = False) -> dict:
    bounds = REGIONS[region]
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    if end_day < start_day:
        raise ValueError("End date precedes start date")
    paths = [(p, SENSORS["modis"]) for p in modis_files] + [
        (p, SENSORS["viirs"]) for p in viirs_files]
    if not paths or not modis_files or not viirs_files:
        raise ValueError("Supply at least one MODIS and one VIIRS S-NPP file")
    for path, _ in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output.mkdir(parents=True, exist_ok=True)
    stage = output / "staging_cells.sqlite"
    if stage.exists() and not overwrite:
        raise FileExistsError(f"{stage} already exists; use --overwrite")
    if stage.exists():
        stage.unlink()
    counts = Counter()
    source_inventory = []
    seen_hashes = set()
    raw_counts = Counter()

    try:
        with sqlite3.connect(stage) as db:
            db.execute("""CREATE TABLE positive_cells (
                date_utc TEXT NOT NULL, grid_x INTEGER NOT NULL,
                grid_y INTEGER NOT NULL, sensor TEXT NOT NULL,
                PRIMARY KEY (date_utc,grid_x,grid_y,sensor)
            ) WITHOUT ROWID""")
            for path, sensor in paths:
                sha = file_sha256(path)
                if sha in seen_hashes:
                    raise ValueError(f"Duplicate input archive (SHA256): {path}")
                seen_hashes.add(sha)
                source_inventory.append({
                    "filename": path.name, "sensor": sensor, "sha256": sha
                })
                for inner_name, chunk in read_chunks(path):
                    normalized = clean_chunk(chunk, sensor, bounds, start_day,
                                             end_day, counts)
                    counts[(sensor, "csv_chunks")] += 1
                    if normalized.empty:
                        continue
                    values = [tuple(row) for row in normalized.drop_duplicates().itertuples(
                        index=False, name=None)]
                    for day, howmany in normalized["date_utc"].value_counts().items():
                        raw_counts[(day, sensor)] += int(howmany)
                    db.executemany(
                        "INSERT OR IGNORE INTO positive_cells VALUES (?,?,?,?)",
                        values,
                    )
                db.commit()

            query = """SELECT date_utc, grid_x, grid_y,
                MAX(CASE WHEN sensor='MODIS' THEN 1 ELSE 0 END) AS modis,
                MAX(CASE WHEN sensor='VIIRS_S_NPP' THEN 1 ELSE 0 END) AS viirs
                FROM positive_cells GROUP BY date_utc,grid_x,grid_y
                ORDER BY date_utc,grid_x,grid_y"""
            daily_cells = defaultdict(lambda: [0, 0, 0, 0])
            grid_path = output / "grid_10km_daily.csv.gz"
            with gzip.open(grid_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date_utc", "grid_x", "grid_y", "modis_positive",
                                 "viirs_positive"])
                for day, gx, gy, mo, vi in db.execute(query):
                    writer.writerow([day, gx, gy, mo, vi])
                    acc = daily_cells[day]
                    acc[0] += 1  # Union cells
                    acc[1] += mo
                    acc[2] += vi
                    acc[3] += int(mo and vi)

        days = []
        for date in pd.date_range(start_day, end_day, freq="D", tz="UTC"):
            day = date.strftime("%Y-%m-%d")
            union, mo, vi, both = daily_cells[day]
            modis_records = raw_counts[(day, "MODIS")]
            viirs_records = raw_counts[(day, "VIIRS_S_NPP")]
            days.append({
                "date_utc": day,
                "year": date.year,
                "modis_record_rows": modis_records,
                "viirs_record_rows": viirs_records,
                "modis_positive_cells": mo,
                "viirs_positive_cells": vi,
                "both_positive_cells": both,
                "union_positive_cells": union,
                "modis_has_positive_records": modis_records > 0,
                "viirs_has_positive_records": viirs_records > 0,
            })
        daily = pd.DataFrame(days)
        daily.to_csv(output / "daily_summary.csv", index=False)
        annual = (daily.groupby("year", as_index=False)
                  .agg(modis_record_rows=("modis_record_rows", "sum"),
                       viirs_record_rows=("viirs_record_rows", "sum"),
                       modis_positive_cell_days=("modis_positive_cells", "sum"),
                       viirs_positive_cell_days=("viirs_positive_cells", "sum"),
                       both_positive_cell_days=("both_positive_cells", "sum"),
                       union_positive_cell_days=("union_positive_cells", "sum"),
                       days_with_modis_positive_records=("modis_has_positive_records", "sum"),
                       days_with_viirs_positive_records=("viirs_has_positive_records", "sum")))
        annual.to_csv(output / "annual_summary.csv", index=False)
        inventory = {
            "region": region, "bounds_west_south_east_north": bounds,
            "utc_period": [start, end],
            "coordinate_system": "EPSG:6933",
            "grid_cell_size_m": GRID_METERS,
            "source_files": source_inventory,
            "row_counts": {sensor: {
                name: counts[(sensor, name)]
                for name in ("input_rows", "excluded_nonvegetation_or_unknown_type",
                             "invalid_rows", "outside_scope_rows",
                             "accepted_rows", "csv_chunks")}
                for sensor in SENSORS.values()},
            "interpretation": (
                "NASA Type-0 presumed vegetation-fire positive hotspot records only, "
                "summarized in UTC-date and "
                "equal-area 10-km cells. Zero records are NOT verified no-fire "
                "observations or proven satellite coverage. Both-positive is "
                "same-cell same-UTC-date co-occurrence, NOT matched acquisition "
                "time, classification accuracy, or independent ground truth. "
                "Do not compare sensors as if equally sensitive."
            ),
            "known_limitations": (
                "Type-0 includes agricultural and other vegetation burning and may "
                "carry source-classification errors; not verified wildfire. "
                "No independent coverage mask, overpass normalization, thermal "
                "background context, or geolocation-anomaly correction is applied."
            ),
        }
        (output / "source_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
        return inventory
    finally:
        if not keep_sqlite and stage.exists():
            stage.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), required=True)
    parser.add_argument("--modis", type=Path, nargs="+", required=True,
                        help="One or more MODIS FIRMS CSV/ZIP files")
    parser.add_argument("--viirs", type=Path, nargs="+", required=True,
                        help="One or more VIIRS S-NPP FIRMS CSV/ZIP files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-sqlite", action="store_true")
    args = parser.parse_args()
    inventory = prepare(args.region, args.modis, args.viirs, args.out,
                        args.start, args.end, args.overwrite, args.keep_sqlite)
    print(json.dumps({"region": inventory["region"],
                      "counts": inventory["row_counts"],
                      "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
