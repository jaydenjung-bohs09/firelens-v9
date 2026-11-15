# FireLens V4 — Instrument availability and provenance

## New
- NASA-sourced historical S-NPP anomaly windows annotated on a daily MODIS/VIIRS hotspot record-count chart
- Distinguishes no VIIRS fire records, NASA FIRMS listed missing-data dates, and partial-day instrument anomaly windows
- Exports daily source-labeled availability/provenance CSV and an exploratory event-window exclusion sensitivity table
- Keeps original FireLens V3 charts and NASA data upload workflow

## Scientific limits
- Historical event flags describe the S-NPP platform in July 2024, not continuous complete satellite coverage.
- 'No hotspot records' is not a statement that there were no fires or that the sensor observed an area with no fires.
- Marking the days of an instrument incident flags a possibly affected calendar day; an incident need not last all day.
- Excluding entire event-window days is exploratory and can bias results.
- For more rigorous coverage normalization, complete orbital footprints and data availability, including zero-fire passes, are needed.

## Sources
- https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html
- https://firms.modaps.eosdis.nasa.gov/api/missing_data/
- https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html

## Installation in Colab
Upload the ZIP. Extract it into `/content`, install `requirements.txt`, and launch `/content/firelens_v3/app.py`. The app still uses `firelens_v3` as its directory for backward compatibility. Upload your MODIS and VIIRS CSVs in the sidebar.
