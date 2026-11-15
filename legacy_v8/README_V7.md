# FireLens V7: Perimeter-constrained time-aware comparison

Keeps all V6 functionality and fixes the V6 7-tab assignment bug. Adds the `Perimeter time matching` tab (tab 8).

## Study design
- Uses the CAL FIRE validated final historical PARK (2024) polygon (not the boundary on an actual day in July).
- Filters hotspot **centroids** within the polygon and optional 0, 1, 2, 5 km buffer.
- Restricts V7 analysis to July 15–31, 2024 (intersection with main global window).
- Compares a 60 km screening circle and official final footprint under the same matching rules.
- Tests 2/5/10 km grids and 1/3/6 hour windows, D/N categories separately, full-date anomaly exclusion.
- Exports region, buffer, temporal, phase, daily, and cell-level tables.
- Explicit scientific limitation: these are **conditional matching rates** among positive hotspot-record cell-date-phases, not true sensor detection accuracy or actual overlap of observing opportunities.

## Starting the app
Upload ZIP in Colab and extract to `/content`, pip install `-r /content/firelens_v7/requirements.txt`, then run `streamlit run /content/firelens_v7/app.py --server.port=8507 --server.address=0.0.0.0 --server.headless=true --server.enableWebsocketCompression=false`.

If you need the prior Colab-specific workaround, add `--server.enableCORS=false --server.enableXsrfProtection=false` temporarily for a private session only, not for public deployment.

## NASA input file names
- MODIS: `fire_archive_M-C61_850130.csv`
- VIIRS S-NPP: `fire_archive_SV-C2_850131.csv`

## Sources
- CAL FIRE historical fire perimeters: https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/content/active_fire/
