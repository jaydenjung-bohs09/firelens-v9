# FireLens — Judge-Readiness Upgrade (V13 Development)

## 90-second demonstration: one coherent story

1. **Global Atlas:** NASA archives show where and when presumed vegetation
   fires were recorded worldwide. Choose one satellite source at a time:
   MODIS (Terra/Aqua, 2000–2025) or VIIRS NOAA-20 (2018–2025).
2. **Park Fire evidence:** Open California Fire Laboratory → Park Fire
   Evidence. The 2024 fire started July 24, but S-NPP VIIRS had a documented
   data interruption beginning July 24 05:28 UTC. The local 2024 archive
   has **zero** VIIRS S-NPP positive records July 24–28 while MODIS positives
   accumulate; VIIRS reports in the rectangle resume July 29.
3. **Evidence inspector:** Show the exact record acquisition times and
   5-km projected groups; moving the maximum time-difference slider changes
   which both-positive groups have a compatible pair. The original V8
   3-hour audit exactly reproduces 316 positive groups, 54 with reports
   from both sensors, and 45 with a time-compatible pair.
4. **AI model audit:** Compare a fixed California ConvLSTM replay with
   "tomorrow=today", any hotspot in the last 7 days, and activity on at
   least 3 of the prior 7 days. Every result uses the same **285** dates
   (March 22–December 31, 2025).

**Intended user and value:** A wildfire intelligence analyst can distinguish
a documented satellite reporting gap from a change in archived hotspot
activity and inspect the original time-aligned evidence before drawing
conclusions about historical patterns.

This is a *retrospective research workflow*, not a real-time emergency
warning system. The project does not claim demonstrated operational
early-warning lead time.

## What the original July 2024 records show

| UTC date | MODIS positives, Park Fire vicinity | VIIRS S-NPP positives |
|---|---:|---:|
| July 24 | 1 | 0 |
| July 25 | 603 | 0 |
| July 26 | 607 | 0 |
| July 27 | 280 | 0 |
| July 28 | 216 | 0 |
| July 29 | 130 | 187 |
| July 30 | 168 | 551 |

The geographic rectangle is 39.2–40.3°N, 122.5–121.0°W. It is not
the historical final perimeter and does not supply independent fire
area labels. NASA FIRMS explicitly lists July 25–28 S-NPP VIIRS data
missing; NASA reports the broader S-NPP anomaly window July 24–29.
Sources:

- https://firms.modaps.eosdis.nasa.gov/api/missing_data/
- https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html
- https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html

**Why the Park Fire is not a critical-period candidate:** The existing
retrospective screening algorithm operates on the *statewide 10-km grid
manifest*, requires both sensors to have positive records on the day,
and excludes known S-NPP anomaly dates. Consequently July 24–28 do
not pass the comparison-quality filter. That is not evidence the fire
was small. The UI now explains this at the screen rather than silently
excluding the dates.

**Why USA–Australia can score 0.706:** The calendar resemblance uses
1 − Jensen–Shannon distance of 12 within-country MODIS monthly shares
in the same UTC calendar months, not climatological seasons or hemisphere
adjustment. Over full years 2001–2025, the U.S. national archive peaks
in August and Australia's in October. Large national totals mix distinct
burning regimes. The app now shows peak months and their gap.

## Forecast: additional experimental evaluation

| Method | Positive-cell F1 | 95% 14-day block-bootstrap CI |
|---|---:|---:|
| Frozen ConvLSTM | 0.4221 | [0.3863, 0.4567] |
| One-day persistence | 0.3831 | [0.3503, 0.4173] |
| Seven-day union | 0.2513 | [0.2312, 0.2717] |
| Activity ≥3 of previous 7 days | 0.3922 | [0.3551, 0.4275] |

All methods are scored on the **same 285 next-day 10-km grid targets**
with the first seven archived test dates excluded because the seven-day
baselines need full history. The numerical CI of model minus one-day
persistence is [0.0297, 0.0478] F1, using 3,000 *circular* 14-day
moving-block bootstrap replicates with random seed 20261115.

This is not untouched independent validation: the 2025 test period
was inspected during an earlier model iteration, and the labels are
positive **NASA FIRMS hotspot-grid reports**, not independently
verified wildfires or nonfires. The bootstrap only addresses one
source of temporal uncertainty under these fixed archived results.
A stronger pre-registered future test would be required to establish
generalization.

Recompute directly from the preserved frozen test archive:

    python v13/forecast_audit.py
    python v13/forecast_audit.py --json

All test archives and original V8 evidence exports are preserved.
The geographic California reference outline is the simplified state
feature from PublicaMundi MappingAPI (us-states.json), projected to
the *exact* EPSG:6933 model-grid origin, with no interpolation of
fire observations. The outline is a geographic reference, not an
official validation mask or daily fire footprint.

## Build and automated tests

    python -m pip install -r requirements.txt pycountry
    python -m unittest discover -s v10 -p 'test_*.py' -v
    python -m unittest discover -s v11 -p 'test_*.py' -v
    python -m unittest discover -s v12 -p 'test_*.py' -v
    python -m unittest discover -s v13 -p 'test_*.py' -v
    python v10/smoke_app.py
    python v12/smoke_inspector.py
    python v13/smoke_storyline.py

Key sources for original data provenance are also in v11/README.md
and the dashboard Methods & Sources panel.
