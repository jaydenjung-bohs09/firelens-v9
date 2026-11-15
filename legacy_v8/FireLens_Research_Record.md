# FireLens — research record for eventual presentation

**Project:** NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots. Solo project.

**Dataset filenames:** MODIS: `fire_archive_M-C61_850130.csv` (July 1–31 2024, California, 4,851 raw records). VIIRS S-NPP: `fire_archive_SV-C2_850131.csv` (same date range/area, 7,770 raw records). Raw detections are not separate fires or burned area.

## Experiment 1: Synthetic grid sensitivity
Synthetic demo, **Aug 1 – Sep 25 2025**. Grid 2/5/10 km occupied cell-days = 788 / 720 / 568; Jaccard = 0.51% / 4.17% / 14.79%. This does NOT represent NASA detections.

## Experiment 2: Real-data grid sensitivity
July 1–31 2024 California. Grid 2 km: MODIS 1787, VIIRS 1478, shared 607, union 2658, Jaccard 22.84%. Grid 5 km: MODIS 732, VIIRS 920, shared 304, union 1348, Jaccard 22.55%. Grid 10 km: MODIS 440, VIIRS 743, shared 199, union 984, Jaccard 20.22%. Grid-size dependence cannot be equated with sensor accuracy.

## Experiment 3: Date-coverage sensitivity
MODIS hotspot records on 31/31 dates; VIIRS hotspot records on 20/31 dates. VIIRS zero-record dates July 10–15 and July 24–28 2024. Shared-detection-dates-only Jaccard at grid 2/5/10 km = 36.65% / 30.40% / 25.25%, vs all-date 22.84% / 22.55% / 20.22%. These are positive record dates, not proof of actual imaging coverage.

NASA MODAPS describes S-NPP GPS/geolocation anomaly July 9–16 and command-data processor anomaly July 24–29. The V4 interruption-window-exclusion Jaccard at 5 km = 28.57% (previously measured). Date exclusion is only sensitivity analysis, not corrected coverage.

## Experiment 4: Park Fire case study
Park Fire ignition July 24 2024. A preliminary 60 km ignition-region exploration showed regional occupied-cell activity rising after July 24, but surrounding non-Park hotspots may contribute. VIIRS absence in the uploaded state file during July 24–28 overlaps NASA-documented disruption. Do not present the file's zero counts as zero fires or proof satellite did not see any locations.

## Experiment 5: Official retrospective CAL FIRE perimeter
The CAL FIRE FRAP **final historical** Park Fire perimeter loaded, labeled PARK 2024, GIS area 429,602.8 acres. This is NOT the perimeter on July 24–31. Within final footprint at 0 km buffer, 3092 raw records = MODIS 2047 + VIIRS 1045 (selected Park study time window). Boundary sensitivity, total records: 0 km 3092; 1 km 3265; 2 km 3274; 5 km 3278. 173/186 added observations within first km; 1 km adds 5.6% vs strict perimeter. These are detections, not burned area or unique fires, and perimeter proximity does not independently prove event identity.

## V6 method
Time-aware detection-matching code in `temporal.py`; retain grid, date and day/night, use UTC acquisition timestamps with 1/3/6 hour tolerance and maximum-cardinality greedy one-to-one matches within each grid-date-phase. Report sensitivity and caveats. No V6 empirical outcome should be inserted in slides until actual exported results are inspected.

## Scientific limitations
FIRMS hotspot CSVs omit non-detection observation opportunities, viewing footprints and cloud quality. MODIS and VIIRS thresholds, spatial resolution and passes differ. Proximity does not prove same physical fire. The final Park Fire footprint is retrospective. Only present experimental claims supported by exported tables, not assumptions of accuracy.

## Sources
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/content/active_fire/
- NASA MODAPS S-NPP outage log: https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html
- CAL FIRE historic perimeters: https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters

## Experiment 6: Real NASA time-aware matching (V6 local verified calculation)
July 1–31 2024 statewide, 5 km grid, same UTC date & same D/N, no outage exclusion. 1-hour: 232/414 co-record cell-date-day/night groups = 56.04% (one-to-one detection pairs 713); 3-hour: 316/414 co-record cell-date-day/night groups = 76.33% (one-to-one detection pairs 1045); 6-hour: 414/414 co-record cell-date-day/night groups = 100.00% (one-to-one detection pairs 1633); 
At 3h/day-night stratification: D: 201/206 = 97.57%, matched pairs 712; N: 115/208 = 55.29%, matched pairs 333; 
IMPORTANT: Conditional matching fractions among positive hotspot-record groups are neither Jaccard union-based agreement nor sensor accuracy. Time-window results are inherently non-decreasing with a wider tolerance; 6h can trivially accept observations hours apart. No evidence of equivalent observation opportunity.


## Experiment 7: V7 geographic and time-aware harmonization (implementation)
V7 adds validated CAL FIRE final-footprint filtering within the July 15–31 2024 record window and compares it to the 60 km screening circle under identical time-aware matching rules. It exports matching for 0,1,2,5 km perimeter buffers, 1/3/6 hour matching tolerances, day/night, daily and per-cell audit. The historical perimeter is the *eventual* fire footprint and not a date-specific fire progression. Results must be entered from the **real** exported V7 comparison tables; no V7 numerical results are asserted before validation in Colab.

## Experiment 8 — Coverage-aware evidence (V8, verified locally)

**Input and scope:** archived California July 1–31, 2024 MODIS `fire_archive_M-C61_850130.csv` and VIIRS S-NPP `fire_archive_SV-C2_850131.csv`. Park Fire case-study window July 15–31, 2024; 5 km EPSG:6933 grid, identical UTC date and day/night phase, 3-hour maximum acquisition-time difference, retrospective unbuffered final Park Fire footprint using the user-exported Park GeoJSON. No date exclusion. This analysis **counts only groups with a positive hotspot record in at least one sensor**, not all locations observed by either satellite.

| Evidence category / metric | Groups | Meaning |
|---|---:|---|
| At least one sensor positive (union of cell/date/phase groups) | 316 | Descriptive denominator, not observation footprint |
| Both sensors positive | 54 | 17.09% of positive-only union |
| Both sensors positive and at least one time-compatible pair | 45 | 83.33% of both-positive groups, or 14.24% of all positive groups |
| MODIS positive, no VIIRS hotspot in that group | 234 | VIIRS coverage/valid non-detection unknown |
| VIIRS positive, no MODIS hotspot in that group | 28 | MODIS coverage/valid non-detection unknown |
| Both positive, but no match within 3 hours | 9 | Sensor observation timing mismatch under selected rule; not necessarily disagreement |
| MODIS-only groups on NASA-listed VIIRS missing-data dates | 208 | Day-level missing-data context; no pixel-level coverage or complete-day outage asserted |

**Checks:** V8 re-computes 3,092 raw detections inside the historical final footprint, and 316 total grouped cell/date/phase rows, reproducing V7's 54 both-positive groups and 45 time-matched groups. Local engine and geometry tests passed; complete interactive Streamlit deployment still needs a live Colab smoke test. This distinction is central: the earlier `83.33%` headline is **conditional** and is not 83.33% of all positive-record groups; across the union it is 14.24%, itself NOT an accuracy metric. Neither fraction is adjusted for actual per-sensor observation opportunities. The previous instrument anomaly dates are retained as documented day-level flags, never used to convert an absent hotspot into a true negative.

For the presentation, do not interpret 'MODIS only' as 'VIIRS missed a fire' or vice versa. The final 2024 footprint is retrospective; the exported GeoJSON is an earlier user-app-produced boundary, not an independently checked live CAL FIRE source capture. Future work requires satellite swath/scan coverage and clear-sky opportunities (including non-detection passes), independent fire event attribution, and uncertainty by date/region before measuring sensitivity or accuracy.
