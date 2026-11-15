# MODVIIR V9 — final submission checklist

## Non-negotiable

- [ ] Verify the actual official 2026 submission form and deadline for your selected event, and make the project page live before last-minute demo polishing.
- [ ] **If participating in the Universal Event and alone:** Official NASA Space Apps GitHub Navigator guidance dated November 12, 2026 says two registered participants minimum for Global Judging. Confirm with event organizers and form an eligible genuine team if required. Local-event rules may differ.
- [ ] If under 18: Review 2026 Space Apps registration and parent/guardian conditions with your event organizer.
- [ ] Supply public working links to a deployed prototype and publicly accessible source/reproduction repository; verify they open signed out.
- [ ] Submit a demo **video ≤30 seconds OR ≤7 slides** if this is still the current format for your event; verify exact official instructions and supplementary-link rules.
- [ ] Disclose AI assistance accurately. Do not imply NASA endorses this prototype.

## Scientific validity

- [ ] Launch `streamlit run app.py` and run a complete browser smoke test (Explore, Investigate, Replay, Methods) on the destination hosting environment.
- [ ] Keep V8's original code unchanged, and preserve its July 2024 inputs as evidence.
- [ ] Park Fire: say 316 positive groups, 54 both-positive groups, 45 time-compatible—not sensor accuracy.
- [ ] The polygon is the **final historic footprint**, NOT a date-specific July progression.
- [ ] The four-year calendar counts **positive grid cell-days**, NOT fires or burned area.
- [ ] Use "no positive records" instead of "no fire" or "sensor observed and missed".
- [ ] Model is retrospective, predicts recorded-hotspot grids, not wildfire ignition or growth; scores are uncalibrated.
- [ ] The 2025 test was examined during earlier development; call it **exploratory**, not independent.
- [ ] Verify selected threshold fixed at 0.675 (validation-selected), no test-based tuning.
- [ ] Model result: test F1 0.4212 vs 0.3818 baseline; 22.1% fewer predicted-only grid cells; all are proxies.

## Differentiation and polish

- [ ] Show the specific question the product answers rather than listing every algorithm.
- [ ] Start with the VIIRS absence interpretation, not a ConvLSTM architecture diagram.
- [ ] In the demo, select July 24 so the anomaly context is visible.
- [ ] Replay actual 2025 grids rather than a simulated prediction animation.
- [ ] Keep narration intelligible; one theme per slide.
- [ ] Leave time for real site submission verification. Do not postpone the project page until the last minutes.

## Three possible submission failures

1. **Validity failure:** Calling record gap a true negative or inference about fire absence.
2. **Relevance failure:** AI prediction overwhelms the official harmonized calendar problem.
3. **Delivery failure:** A locally running notebook exists but judges cannot access the deployed app, source, or demo—or the team is ineligible for the judging track.

## Sources for requirements

- Official NASA Space Apps 2026: https://www.spaceappschallenge.org/2026/
- NASA GitHub Navigator clarification (Nov 12, 2026): https://github.com/nasa/spaceapps/discussions/1947
- 2026 legal terms, including minor participation provisions: https://www.spaceappschallenge.org/legal/
