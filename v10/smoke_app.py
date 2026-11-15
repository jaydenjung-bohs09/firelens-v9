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
if any(s.key=="v11_global_year" for s in app.get("select_slider")):
    raise AssertionError("Raw year control should be hidden by default")
raw_view=[item for item in app.checkbox if item.key=="v15_show_raw_sensor_atlas"]
if len(raw_view)!=1:
    raise AssertionError("Global raw-sensor toggle is missing")
raw_view[0].set_value(True)
app.run()
assert_clean("Raw atlas toggle")
if not any(s.key=="v11_global_year" for s in app.get("select_slider")):
    raise AssertionError("Global archive year control missing when raw view enabled")

# The secondary global sensor has a shorter historical era.
nav("v11_global_sensor").set_value("VIIRS_NOAA20")
app.run()
assert_clean("Global NOAA-20")
global_years=[v for v in app.get("select_slider") if v.key=="v11_global_year"]
if not global_years:
    raise AssertionError("Global NOAA-20 year selector missing")
global_years[0].set_value(2025)
app.run()
assert_clean("Global NOAA-20 year 2025")

nav("v11_workspace").set_value("02 · COUNTRY COMPARISON")
app.run()
assert_clean("Country comparison")
if not any(m.key=="v11_compare_countries" for m in app.multiselect):
    raise AssertionError("Country selection UI not found")

nav("v11_compare_sensor").set_value("VIIRS_NOAA20")
app.run()
assert_clean("Country comparison with NOAA-20")
compare_years=[v for v in app.get("select_slider") if v.key=="v11_overlap_year"]
if not compare_years:
    raise AssertionError("Two-sensor perspective year control missing")
compare_years[0].set_value(2025)
app.run()
assert_clean("Two-sensor perspective 2025")

nav("v11_workspace").set_value("03 · CALIFORNIA FIRE LABORATORY")
app.run()
assert_clean("California laboratory")
sections=[
    "02  INVESTIGATE EVIDENCE",
    "03  REPLAY THE MODEL",
    "01  EXPLORE HISTORY",
    "METHODS & SOURCES"
]
for section in sections:
    nav("v11_california_tool").set_value(section)
    app.run()
    assert_clean("California subsection: "+section)

# The Amazon field study moved to the optional global research expander.
nav("v11_workspace").set_value("01 · GLOBAL ATLAS")
app.run()
assert_clean("Global after California")
# Streamlit clears state of hidden widgets when users switch workspaces.
raw_return=[c for c in app.checkbox if c.key=="v15_show_raw_sensor_atlas"]
if len(raw_return)!=1:
    raise AssertionError("Raw-sensor toggle missing on return to global")
raw_return[0].set_value(True)
app.run()
assert_clean("Reopen raw source charts")
amazon=[c for c in app.checkbox if c.key=="v13_open_amazon"]
if len(amazon)!=1:
    raise AssertionError("Global Amazon research toggle missing")
amazon[0].set_value(True)
app.run()
assert_clean("Optional Amazon study")
years=[x for x in app.selectbox if x.key=="amazon_year"]
if not years:
    raise AssertionError("Amazon year selector missing from optional panel")
years[0].set_value(2025)
app.run()
assert_clean("Amazon 2025")
print("PASS: Global + Country + 4 California tools + optional Amazon 2025")
