"""FireLens V11 full Streamlit smoke test: sidebar workspaces and lab tools."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest

app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=120)
def assert_clean(label):
    if len(app.exception):
        raise AssertionError(label+" exception(s): "+str([e.message for e in app.exception]))
def nav(key):
    matches=[r for r in app.radio if r.key==key]
    if len(matches)!=1:
        raise AssertionError(f"Expected one {key}; got {len(matches)}")
    return matches[0]

app.run()
assert_clean("Global startup")
if nav("v11_workspace").value!="01 · GLOBAL ATLAS":
    raise AssertionError("Global Atlas must be landing page")
if not any(s.key=="v11_global_year" for s in app.get("select_slider")):
    raise AssertionError("Global archive year control not found")

nav("v11_workspace").set_value("02 · COUNTRY COMPARISON")
app.run()
assert_clean("Country comparison")
if not any(m.key=="v11_compare_countries" for m in app.multiselect):
    raise AssertionError("Country selection UI not found")

nav("v11_workspace").set_value("03 · CALIFORNIA FIRE LABORATORY")
app.run()
assert_clean("California laboratory")
sections=[
    "01  EXPLORE HISTORY",
    "02  INVESTIGATE EVIDENCE",
    "03  REPLAY THE MODEL",
    "04  AMAZON 2022–2025",
    "METHODS & SOURCES"
]
for section in sections:
    nav("v11_california_tool").set_value(section)
    app.run()
    assert_clean("California subsection: "+section)

nav("v11_california_tool").set_value("04  AMAZON 2022–2025")
app.run()
assert_clean("Amazon load")
years=[x for x in app.selectbox if x.key=="amazon_year"]
if not years:
    raise AssertionError("Amazon year selector not found")
years[0].set_value(2025)
app.run()
assert_clean("Amazon 2025")
print("PASS: Global + Country + all 5 California research tools + Amazon 2025")
