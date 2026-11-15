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
navs[0].set_value("04  AMAZON 2022–2025")
app.run()
if len(app.exception):
    raise AssertionError("V10 Amazon panel raised: " + "\n".join(str(e.message) for e in app.exception))
print("PASS: V9.1 startup and Amazon V10 panel rendered without exceptions.")
