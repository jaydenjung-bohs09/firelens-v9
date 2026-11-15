# V9 QA report

Verified offline with actual uploaded archives on 2026-11-14 (Pacific) / 2026-11-15 (UTC).

- Original July 2024 MODIS+VIIRS records: 4,851+7,770 = 12,621.
- Exact V8 perimeter-filtered July 15–31 detections: 3,092.
- Exact V8 5 km, three-hour positive evidence groups: 316 union / 54 both-positive / 45 time-compatible.
- Model frozen 2025 scores re-evaluated with original float16 threshold semantics: TP 3,581 / FP 4,363 / FN 5,480; F1 0.421170; IoU 0.266761.
- Baseline frozen test metrics: TP 3,460 / FP 5,603 / FN 5,601; F1 0.381814; IoU 0.235952.
- Manifest: 1,454 sequences, 870/292/292 chronological train/validation/test.
- Original `legacy_v8` copy retained without changes.
- `tests/test_integrity.py` passed.
- `tests/smoke_without_streamlit.py` completed seven Plotly figure creations with a simulated UI/control stub.
- Seven-slide deck generated and converted to PDF; rendered-page montage inspected for overlaps and adjusted.

**Important unverified item:** Could not start a real Streamlit server in the local authoring environment because the Streamlit package was not installed and external package-network access was unavailable. Thus the final UI requires a Google Colab/Streamlit Cloud/browser smoke test. This package is not represented as a deployed product.

**Build constraint:** Four-year MODIS/VIIRS source ZIP archives were not included in the conversation upload. The full raw multi-year data are therefore not bundled; the four-year calendar is reproduced from the submitted sequence manifest, and the replay is from the frozen model's 292-day score archive. The original July 2024 files and perimeter export are included.
