"""Judge journey smoke test: global → Park Fire evidence → AI comparison."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest

def find(app,kind,key):
    values=[x for x in app.get(kind) if x.key==key]
    if len(values)!=1:
        raise AssertionError(f"Expected {kind} {key}: got {len(values)}")
    return values[0]

def clean(app,label):
    if app.exception:
        raise AssertionError(f"{label}: "+", ".join(str(x.message) for x in app.exception))

app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=120).run()
clean(app,"Initial world")
find(app,"button","v13_global_story").click().run()
clean(app,"Navigate to California")
assert find(app,"radio","v11_workspace").value=="03 · CALIFORNIA FIRE LABORATORY"
assert find(app,"radio","v11_california_tool").value=="02  INVESTIGATE EVIDENCE"
find(app,"button","v13_park_to_model").click().run()
clean(app,"Navigate to model")
assert find(app,"radio","v11_california_tool").value=="03  REPLAY THE MODEL"
if not any("ConvLSTM" in str(x.label)
           for x in app.get("metric")):
    raise AssertionError("Forecast metrics missing")
print("PASS: guided global → Park Fire → AI evaluation navigation")
