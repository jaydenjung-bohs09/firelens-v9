# FireLens V11 — Global Atlas → Country Comparison → California Laboratory

**NASA International Space Apps Challenge 2026**  
Challenge: Harmonization of MODIS and VIIRS Hot Spots  
Official challenge: https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/

## Three left-side workspaces

1. **Global Fire Atlas** — sensor-selectable choropleth, yearly archive trends, and a monthly burning-activity calendar across 2000–2025.
2. **Country Comparison** — select 2–5 countries, use one sensor over the same period, inspect absolute and indexed trends, monthly seasonal fingerprints, Jensen–Shannon resemblance, historical monthly departures, and separate-sensor geographic-share comparisons.
3. **California Fire Laboratory** — preserved California 2022–2025 historical calendar, real July 2024 area-of-interest exploration (including a Brea/North Orange County rectangle), Park Fire MODIS–S-NPP evidence matching, frozen ConvLSTM model replay, Amazon regional evidence case study, and methods/provenance.

The sidebar is the primary navigation. Global and country views load only the compact historical table. California model arrays load **only inside the California Laboratory** to keep worldwide browsing responsive.

## Real worldwide source inventory

| Original NASA source | Coverage | Archive ZIPs | Positive type-0 archive records |
|---|---:|---:|---:|
| MODIS Collection 6.1, Terra/Aqua | 2000–2025 | 26 | 107,673,642 |
| VIIRS NOAA-20 / JPSS-1, 375 m | 2018–2025 | 8 | 145,241,663 |
| **Combined** | Different sensor eras | **34** | **252,915,305** |

The input archives contain **267,356,689 raw thermal-anomaly rows**. Filtering to NASA FIRMS **type 0 presumed vegetation fires** yields **252,915,305** positive-record rows, aggregated into **7,033 source country-year entries** across **238 country/territory names**. The original user-provided NASA yearly country archives total multiple gigabytes; we do **not** commit them to the app.

Files in this repository:

- [global_country_year.csv](processed/global/global_country_year.csv) — compact CSV: source sensor, archive year, country/ISO3, raw and type-0 counts, active recorded days, acquisition day/night counts and 12 monthly type-0 counts
- [global_manifest.json](processed/global/global_manifest.json) — validated aggregate totals and interpretation constraints
- [archive_checksums.csv](processed/global/archive_checksums.csv) — SHA256 hashes, raw and vegetation rows for each of the original 34 ZIPs
- [build_global_atlas.py](build_global_atlas.py) — streaming reproducibility script with country-name mappings
- [world.py](world.py) — independent global and country dashboards

These are archived observations, **not counts of distinct wildfires**, hectares burned, verified satellite non-detections, or validated sensor sensitivity. Type-0 includes agricultural and prescribed burning and may contain classification errors.

### Different starting years matter

MODIS begins in **November 2000**; NOAA-20 VIIRS begins in **April 2018**. The two first archive years are incomplete. We avoid combining their raw counts into one universal activity rate or claiming that their relative record totals measure detection performance. Calendar charts normally show one sensor at a time.

### Scientific logic for exploratory features

**Seasonal fingerprint.** For a given country, satellite product, and selected period, sum its 12 monthly positive-record counts and divide each month by the country-period total. All rows sum to 100%. This compares *seasonal distributions* rather than unequal absolute counts.

**Seasonal resemblance.** For two countries' 12-month probability distributions, the score is

    resemblance = 1 - Jensen–Shannon distance (base 2)

where 1 means identical monthly proportions and 0 indicates no monthly overlap. This is descriptive; neither weather attribution nor real-world fire spread has been demonstrated.

**Retrospective unusual-month screen.** For a selected country and year, use only **earlier years' same-month archived counts**, excluding partially covered starting years. Require at least 3 full-year peers. A candidate exceeds the earlier-year 90th percentile, is ≥1.5× the earlier median, and has ≥100 records. This is **not** a statistical significance test, prospective fire prediction, validated anomaly frequency, or emergency alert.

**Two-satellite geographic perspective.** Within a selected archive year, calculate each selected country's percentage of that **sensor's own total world type-0 records**. Differences in the geographic shares are not MODIS–VIIRS detection accuracy, pixel-level agreement, nor fire severity. This is a compositional research lens, not full physical sensor harmonization.

**Genuine sensor harmonization case study** remains the California Park Fire evidence laboratory: 316 positive record groups, 54 both-sensor positive groups and 45 time-compatible groups under the specified UTC date/day-night/3-hour matching rule; no independently verified observed negatives. The existing California ConvLSTM frozen replay uses recorded-hotspot next-day labels, with exploratory 2025 F1 0.4212 and a persistence F1 baseline of 0.3818. **The model has not been tested globally.**

## Rebuild and test

The app:

    python -m pip install -r requirements.txt
    streamlit run app.py

For original archives (keep **all 34** source filenames in the same directory):

    python -m pip install pycountry
    python v11/build_global_atlas.py --input "/path/to/nasa_zips" --output "/path/to/audited_output"

This recreates country-year counts and the full source-hash manifest. It intentionally checks the known 7,033 rows and 252,915,305 type-0 records; if the archive version changes, review and update those audited expectations rather than quietly substituting files.

Automated checks:

    python -m unittest discover -s v11 -p "test_*.py" -v
    python -m unittest discover -s v10 -p "test_*.py" -v
    python v10/smoke_app.py

The Streamlit smoke test renders Global Atlas, Country Comparison, every California research subtool, and the Amazon 2025 controls. The program is a **retrospective research prototype** and must never be used as an operational fire alarm or to direct emergency response.

## Why this fits NASA's challenge

- **Impact:** a worldwide historical navigation layer with country and local views
- **Creativity:** relative seasonal fingerprints, country resemblance and evidence-quality context
- **Validity:** source SHA256, explicit sensor-era boundaries, chronological reference sets, reproducible tests, uninflated scientific conclusions
- **Relevance:** historical active-hotspot calendar and detailed MODIS–VIIRS time/spatial evidence investigation
- **Presentation:** a guided journey from the planet to individual countries and the California case study

NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/download/
