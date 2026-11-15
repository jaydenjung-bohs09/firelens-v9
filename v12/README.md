# FireLens V12 — California Satellite Evidence Inspector

A record-level, time-aware Park Fire case study inside
**California Fire Laboratory → Sensor Harmonization**.

## What it adds

- Per-date and day/night grouping in **5-km projected equal-area**
  EPSG:6933 grid cells.
- Side-by-side MODIS and VIIRS S-NPP **positive archive records** and
  actual recorded acquisition times in UTC.
- Recomputed maximum-cardinality, one-to-one pair matching when the user
  changes the allowable time difference from 0 through 720 minutes.
- Status categories for both-positive/time-compatible,
  both-positive/time-separated, MODIS-only with documented S-NPP flags,
  MODIS-only with unknown VIIRS opportunity, or VIIRS-only with unknown
  MODIS opportunity.
- Options to exclude known S-NPP anomaly dates from **headline rates**
  (the inspector still shows them) and to restrict to NASA type 0,
  **presumed vegetation-fire** records.
- A sensitivity plot showing how time tolerance affects the number of
  both-positive groups with a possible acquisition-time pair.
- Downloadable complete group audits and the selected cell's source records.

## Reproduction and data limitations

**Original NASA inputs and exports, untouched:**

- MODIS: data/historic/fire_archive_M-C61_850130.csv
- VIIRS S-NPP: data/historic/fire_archive_SV-C2_850131.csv
- Original final-footprint subset: data/historic/park_hotspot_records.csv
- Original reference table: data/historic/park_evidence_groups.csv
- NASA-disruption context: data/historic/july_2024_sensor_status.csv

The final Park Fire perimeter geometry is a **retrospective**, user-exported
historical perimeter, not daily fire spread. The 3,092 Park Fire positive
records are from **July 25–31, 2024** in the supplied footprint, under
the original larger July 15–31 study window.

The 3-hour configuration reproduces frozen V8 **exactly**:
316 positive-report groups, 54 with positives in both sensors,
45 with at least one time-compatible pair, 149 one-to-one pairs.

This is NOT:
- confirmed co-observation by both sensors for every cell,
- calibrated detection sensitivity or sensor accuracy,
- a classification of unreported cells as no-fire,
- an estimate of burned area or distinct wildfire counts,
- proof that paired record centroids correspond to the same fire,
- a real-time wildfire warning or geolocation-corrected product.

**Zero positive reports from a satellite does not mean the satellite
imaged that cell and found no fire.** Documented S-NPP disruptions make
some record gaps especially difficult to interpret.

The original archive has eight NASA type-2 thermal anomaly records among
the 3,092. Selecting type 0 changes the group universe, so the 316/54/45
reference check applies only when that filter is off.

## How to test

Run the test commands from the repository root:

    python -m unittest discover -s v12 -p 'test_*.py' -v
    python v12/smoke_inspector.py

The unit tests compare all 316 cell/date/phase keys and group-level
record and matched-pair counts against the unchanged frozen V8 export,
then test window monotonicity, flags and optional type-0 filtering.
AppTest exercises slider, date, phase and evidence-quality controls.

## NASA Space Apps relevance

The inspector exposes the evidence behind an *apparent disagreement*:
positive-record co-occurrence, temporal alignment, documented platform
anomalies, and unknown satellite observation opportunities. It is a
carefully bounded positive-record harmonization investigation, not
a claim to have achieved physical sensor cross-calibration.
