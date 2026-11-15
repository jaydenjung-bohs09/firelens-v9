"""Interactive Streamlit UI smoke tests for the Park Fire Evidence Inspector."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest


def control(app, kind, key):
    matches = [item for item in app.get(kind) if item.key == key]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {kind} with key={key}, got {len(matches)}")
    return matches[0]


def assert_no_exceptions(app, description):
    if app.exception:
        raise AssertionError(
            description + ": " + " | ".join(str(x.message) for x in app.exception)
        )


app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
app.run()
assert_no_exceptions(app, "Initial global view")
control(app, "radio", "v11_workspace").set_value(
    "03 · CALIFORNIA FIRE LABORATORY"
)
app.run()
assert_no_exceptions(app, "California")
control(app, "radio", "v11_california_tool").set_value(
    "02  INVESTIGATE EVIDENCE"
)
app.run()
assert_no_exceptions(app, "Inspector initial")
control(app, "slider", "v12_time_tolerance").set_value(60)
app.run()
assert_no_exceptions(app, "60-minute recalculation")
control(app, "checkbox", "v12_type0_only").set_value(True)
app.run()
assert_no_exceptions(app, "NASA type-0 filtered recalculation")
control(app, "checkbox", "v12_exclude_flags").set_value(True)
app.run()
assert_no_exceptions(app, "Anomaly-screened summary")
control(app, "radio", "v12_phase").set_value("Night")
app.run()
assert_no_exceptions(app, "Night-only selection")
control(app, "selectbox", "v12_date").set_value("2024-07-31")
app.run()
assert_no_exceptions(app, "Selected July 31")
control(app, "slider", "v12_time_tolerance").set_value(180)
app.run()
assert_no_exceptions(app, "Restore three-hour reference")
print("PASS: evidence inspector rerenders across tolerance, type, anomaly, phase and date controls")
