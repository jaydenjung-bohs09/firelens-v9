# FireLens V5 — official historical Park Fire polygon

Adds an official CAL FIRE FRAP historical perimeter filter to the V4 dashboard. The app pulls GeoJSON from the public ArcGIS historical perimeter feature service on demand (internet required), verifies that the fire is named Park and dates to 2024, and checks the polygon is near the known ignition site. No fabricated boundary ships with this package.

**Scientific note:** The polygon is the *final historical footprint* (released after the fire). It is not the fire boundary during July 2024. Hotspots inside it are not necessarily detections from the Park Fire and must not be interpreted as burned area or a fire progression. The 0/1/2/5 km buffer is for geolocation-sensitivity testing only.

Run in Colab with `streamlit run app.py`, upload both FIRMS CSVs, select July 1–31, 2024, and go to **Park Fire case study → Official historical perimeter → Load**. If ArcGIS is blocked, obtain the Park Fire GeoJSON from CAL FIRE and upload it to the same section; it must retain `FIRE_NAME`, `YEAR_`, and polygon coordinates. Geospatial dependencies added: shapely, requests.

Source: https://services3.arcgis.com/CSK2qijkklTsqQyF/ArcGIS/rest/services/CaliforniaFirePerimeters2024/FeatureServer/0
