# FireLens — Evidence Before Assumptions

**Challenge:** Harmonization of MODIS and VIIRS Hot Spots — NASA Space Apps Challenge 2026

### What problem are we addressing?

NASA's MODIS and VIIRS sensors report active-fire hotspots with different spatial resolutions, acquisition times, detection characteristics and periods of disrupted record availability. Placing the two positive-report archives side by side can create misleading conclusions: a missing report is **not** necessarily proof of no fire. Researchers and fire-monitoring teams need to understand when satellite records provide positive evidence, when reports are temporally compatible, and when the evidence remains incomplete.

### The solution

FireLens is an interactive California research prototype with three connected experiences. **Explore** presents a four-year calendar of daily 10 km grid-cell hotspot-record activity, with flags when one or both sensors have no records that day. **Investigate** reconstructs the July 2024 Park Fire case using equal-area grid cells, day/night distinction, three-hour temporal matching, and documented S-NPP disruptions, without calling one-sided records sensor failures. **Replay** displays a trained ConvLSTM's saved next-day **recorded-hotspot** scores and compares them against actual archived reports and a persistence baseline.

### Empirical evidence

In the retrospective final Park Fire footprint for July 15–31, 2024, V8 found 316 positive-only cell/date/day-night groups: 54 groups had records from both sensors, and 45 of those were time-compatible within three hours. These are record-based comparisons, **not accuracy estimates**.

A separate ConvLSTM trained on California NASA FIRMS data from 2022–2025 achieved an exploratory 2025 test F1 of **0.4212**, versus **0.3818** for a last-day persistence baseline (10.3% relative improvement), with 22.1% fewer predicted-only grid cells. The model predicts future **detected hotspot records**, not ignition or verified fire spread. The test period had been examined during earlier development, so the result is exploratory rather than independent final confirmation.

### Impact, boundaries, and next steps

The immediate contribution is **interpretable scientific evidence**, with an experimental extension for retrospective forecast exploration. FireLens is not connected to emergency alerts and is not appropriate for operational fire decisions. Its next scientific steps are pixel-level overpass/clear-sky opportunity, better cross-sensor calibration, independent event-based validation, and expansion beyond California.

### Resources and code

NASA FIRMS MODIS C6.1 and VIIRS S-NPP archives; NASA MODAPS disruption reports; CAL FIRE historical Park Fire perimeter. The V8 code, V9 code, model checkpoint, frozen prediction arrays, integrity checks and supporting exports are provided in the repository/package.

### AI assistance disclosure

The project uses a ConvLSTM machine-learning model. ChatGPT assisted with software generation, debugging, scientific writing, and presentation structuring; its outputs and scientific statements were reviewed against archived files and numerical integrity checks. Disclose any other AI-assisted assets or tools used in the final submitted materials, consistent with NASA Space Apps guidance.
