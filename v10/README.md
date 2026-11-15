# FireLens V10 — Regional NASA archive preparation (staging branch)

This is the **data foundation**, not a deployed V10 dashboard. It is designed to
accept future NASA FIRMS historical archives without modifying the working V9.1.

## First target

- Region: Amazon study rectangle, W/S/E/N \`-74,-15,-48,2\`
- Period: **2022-01-01 to 2025-12-31** (UTC)
- Historical products: **MODIS Collection 6.1** and **VIIRS S-NPP 375 m**
- Formats: original NASA CSV or ZIP; files can cover multiple calendar years
- NASA request page: https://firms.modaps.eosdis.nasa.gov/download/

Keep the two sensor groups separate when passing them to the importer.
The script accepts **one file or many per sensor** and splits data into
daily and annual summaries automatically. It validates optional \`instrument\`
values against the sensor selected at the command line.

## Run after the data arrives

First install the existing repository requirements:

\`\`\`bash
python -m pip install -r requirements.txt
\`\`\`

For one multi-year file (or ZIP) per sensor:

\`\`\`bash
python v10/prepare_region.py \
  --region amazon \
  --modis "/path/to/NASA_MODIS_file.zip" \
  --viirs "/path/to/NASA_VIIRS_SNPP_file.zip" \
  --out "v10/processed/amazon"
\`\`\`

For four annual files per sensor, append additional file paths after each
flag, ending each list before the next flag. Existing outputs require
\`--overwrite\` to regenerate.

## Real NASA Amazon data received

The source archive ZIPs `DL_FIRE_M-C61_850616.zip` and `DL_FIRE_SV-C2_850617.zip` cover 2022–2025 and were processed separately. The observed total is **4,404,388 raw records**; after filtering to type-0 presumed vegetation-fire records, **4,386,124** remained. This is **917,907 positive union 10-km cell-days**, not unique wildfire events or burned area.

The repository bundles lightweight derived `processed/amazon/` daily, annual, spatial-1° and provenance files, not the original 75 MB of archive ZIPs. The exact source SHA-256 values are recorded in `source_inventory.json`.

## Generated artifacts

| Artifact | Purpose |
|---|---|
| \`daily_summary.csv\` | 2022–2025 daily recorded hotspot counts, sensor-specific and union grid cells |
| \`annual_summary.csv\` | Yearly positive-record counts and positive cell-days |
| \`grid_10km_daily.csv.gz\` | GIS-ready 10-km same-UTC-day grid, both sensors represented separately |
| \`source_inventory.json\` | File SHA-256, provenance, observed counts, preprocessing scope and limitations |

Only source `type=0` presumed vegetation-fire records are processed for the regional view; type-2 and other thermal anomalies are counted in provenance but excluded. This does **not** identify distinct wildfires; agricultural and managed burning may remain. NASA's VIIRS archive uses `instrument=SNPP`, which is recognized as VIIRS S-NPP.

All dates are **UTC**. Coordinates use **EPSG:6933** and **10-km projected
equal-area bins**, for exploratory spatial comparisons. Cells are NOT burned
area polygons. Identical cell/date positives do NOT establish simultaneous
overpasses, independent detection sensitivity, satellite accuracy, or
ground-truth confirmed fire. Zero records do NOT establish coverage or
absence of fire. VIIRS S-NPP anomaly flags/quality masks must be incorporated
before sensor-normalized comparisons.

Processing uses pandas chunks and an on-disk SQLite staging table for
bounded-memory aggregation. It does not require keeping the entire raw
dataset in Streamlit's memory. Processing runtime and disk space depend on
archive size; global-scale archives will require a stronger backend.

**Scientific scope:** The regional data foundation is descriptive only.
The existing California ConvLSTM must not be presented as trained or
validated on Amazon data. Future regional forecasting requires independent
time/region-separated evaluation and meaningful baselines.

## Test with synthetic files before real data arrives

\`\`\`bash
python -m unittest discover -s v10 -p "test_*.py" -v
\`\`\`

Tests cover ZIP parsing, sensor label validation, file duplication, raw-record
counts vs unique occupied grid cells, cross-sensor cell co-occurrence, the
yearly totals, and zero-positive-record days.

## Staging dashboard

On branch `v10-data-foundation`, run `streamlit run app.py` and choose **04 AMAZON 2022–2025**. The panel provides an archive-wide record count, annual trends, monthly seasonality, daily sensor-positive indicators, a spatial 1° summary from 10-km equal-area cell-days, and retrospective critical-period screening.

The displayed spatial tiles are coarse index bins, not true fire perimeters. Model replay continues to use only the original California trained model.

## Staging and deployment

Branch: \`v10-data-foundation\`. The live Streamlit app on \`main\`
remains unchanged. The data files will be stored outside GitHub until
they are cleaned, tested and reduced to deployment-safe summaries.

Later integration steps:
1. Import and audit real NASA archive counts.
2. Validate region/date filters and source product metadata.
3. Add regional selector and timeline to a staging dashboard.
4. Verify evidence flags, map rendering, load time and mobile usability.
5. Deploy only after the existing V9.1 functionality still passes tests.
