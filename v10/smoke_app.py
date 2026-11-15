"""Smoke-test the complete V9.1 shell and new V10 Amazon panel with Streamlit AppTest."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

app=AppTest.from_file(str(ROOT / "app.py"),default_timeout=90)
app.run()
if len(app.exception):
    raise AssertionError("App startup raised: " + "\n".join(str(e.message) for e in app.exception))
navs=[r for r in app.radio if r.label=="Choose section"]
if not navs:
    raise AssertionError("Section radio missing from app")
for section in [
    "01  EXPLORE HISTORY", "02  INVESTIGATE EVIDENCE",
    "03  REPLAY THE MODEL", "04  AMAZON 2022–2025",
    "METHODS & SOURCES",
]:
    app.radio[0].set_value(section)
    app.run()
    if len(app.exception):
        raise AssertionError(section + " raised: " + "\n".join(
            str(e.message) for e in app.exception))

app.radio[0].set_value("04  AMAZON 2022–2025")
app.run()
amazon_year = [x for x in app.selectbox if x.key == "amazon_year"]
if not amazon_year:
    raise AssertionError("Missing Amazon year selector")
amazon_year[0].set_value(2025)
app.run()
if len(app.exception):
    raise AssertionError("Amazon year switch raised: " + "\n".join(
        str(e.message) for e in app.exception))
print("PASS: all five sections and Amazon 2025 selector rendered without exceptions.")
