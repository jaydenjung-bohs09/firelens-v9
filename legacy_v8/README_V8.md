# FireLens V8 — Coverage-aware evidence

Adds a ninth dashboard tab: **Coverage-aware evidence**. Displays three distinct denominators (both-positive over union, time-compatible over both-positive, and time-compatible over union), classifies one-sided hotspot evidence without implying the absence of satellite observation, shows date-level documented S-NPP anomalies and NASA-listed missing dates, and exports three tables.

The bundled `park_fire_verified_historical_footprint_USER_EXPORT.geojson` is an export from the user's earlier V7 CAL FIRE-perimeter workflow, validated for PARK/2024 and approximate location by the application, but not an independent live CAL FIRE layer verification. The perimeter is the final retrospective footprint, never the July day-specific fire boundary.

Run on Google Colab after uploading ZIP and extracting to `/content`, `pip install -r /content/firelens_v8/requirements.txt`, then `streamlit run /content/firelens_v8/app.py --server.port 8508 --server.headless true --server.address 0.0.0.0 --server.enableWebsocketCompression false`. For private Colab iframe compatibility, you may temporarily add disableCORS/XSRF flags; do not deploy publicly with security protections disabled.

Input filenames: MODIS `fire_archive_M-C61_850130.csv`; VIIRS `fire_archive_SV-C2_850131.csv`. Use July 1–31 2024, 5 km grid and 3-hour tolerance as a baseline.

**Interpretation:** zero FIRMS hotspot records ≠ zero fires or verified no observation; documented instrument windows do not prove complete missing daily coverage. These analyses cannot estimate true accuracy without satellite observation-opportunity footprints.
