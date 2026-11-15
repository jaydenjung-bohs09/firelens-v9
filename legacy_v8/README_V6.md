# FireLens V6 — Time-aware detection matching

This is the **full V5 project plus V6**, not a patch. The original CAL FIRE perimeter filter, scientific audit, synthetic experiment, availability timeline and main dashboard remain included.

## Install in Colab

Upload the V6 ZIP through `google.colab.files.upload()`, extract to `/content`, run `%pip install -q -r /content/firelens_v6/requirements.txt`, start `streamlit run /content/firelens_v6/app.py --server.port=8506 --server.address=0.0.0.0 --server.headless=true --server.enableWebsocketCompression=false`, and expose with `output.serve_kernel_port_as_iframe(8506, height=850)`.

Real files (do not confuse them): MODIS `fire_archive_M-C61_850130.csv`; VIIRS `fire_archive_SV-C2_850131.csv`.

## V6 scientific method

- Parse NASA FIRMS `acq_date`, `acq_time` (UTC HHMM), and `daynight`.
- Group by a projected EPSG:6933 cell, UTC date and day/night; compare reported hotspot acquisition timestamps.
- Within each cell-date-phase group, use a sorted two-pointer one-to-one matching within a configurable 1/3/6 h tolerance.
- Report both-record cell-day-phase groups, groups with ≥1 time-match, and unique matched detection pairs.
- Time-match fraction = `groups with ≥1 time-match / groups with records from both products`, NOT recall, precision, agreement of independent full satellite coverages or probability of same physical fire.
- Days in S-NPP anomaly windows can be excluded as a conservative *sensitivity test*, not as a coverage normalization.
- Data exports: time-window sensitivity, day/night analysis, daily matched-cell table, and per-cell audit.

## Important limitations

- Two satellite records are not independent fire identities.
- Same cell and time proximity cannot prove identical physical wildfire; strict grid cells can miss real nearby associations.
- Same UTC date constraint misses records across midnight.
- Detection CSV contains only *positive detections*, not passes with zero fires, cloud coverage, or sensor viewable footprint.
- VIIRS S-NPP documented July 2024 interruptions materially affect archive comparisons.
- The Park Fire historical final polygon is retrospective, not a July 2024 daily growth map.
- This prototype is for research and **not** an operational emergency alert.

## Inputs and evidence

NASA FIRMS historical archives: https://firms.modaps.eosdis.nasa.gov/download/

NASA Suomi NPP outage log: https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html
