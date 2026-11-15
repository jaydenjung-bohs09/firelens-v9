"""Present tested Amazon daily calibration without altering existing archive counts."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from v15.amazon_tier2 import ROOT

RESULT=ROOT/"v15/amazon_tier2_results.json"

def render_tier2():
    d=json.loads(RESULT.read_text())
    st.subheader("Amazon: should we compare records or positive cells?")
    st.caption(
        "Complete source-calendar years 2022–2024 train three MODIS-to-"
        "VIIRS S-NPP mapping methods. All 365 days of 2025 are held out. "
        "The 14 flagged S-NPP disruption dates in 2024 are excluded "
        "from training. No missing satellite opportunity is inferred."
    )
    labels={"records":"Raw hotspot records",
            "positive_cells":"10-km positive cells"}
    method_labels={"none":"Unadjusted MODIS","ratio":"Monthly ratio",
                   "log":"Log regression + month"}
    table=[]
    for unit,v in d["units"].items():
        for method,stats in v["methods"].items():
            table.append({
                "Measurement":labels[unit],
                "Conversion":method_labels[method],
                "Median absolute error (%)":round(stats["median_abs_pct_error"],3),
                "Median bias (%)":round(stats["median_bias_pct"],3),
                "Spearman ρ":round(stats["spearman"],3),
                "Held-out days":stats["n_evaluated"],
            })
    st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
    records=d["units"]["records"]["methods"]["ratio"]["median_abs_pct_error"]
    cells=d["units"]["positive_cells"]["methods"]["ratio"]["median_abs_pct_error"]
    if records<=cells:
        message="Raw hotspot-record counts had the lower held-out median absolute percentage error"
    else:
        message="Positive-cell counts had the lower held-out median absolute percentage error"
    st.write(
        f"**{message}** ({records:.2f}% for records versus "
        f"{cells:.2f}% for positive 10-km cells) using the winning "
        "monthly-ratio method for each measurement."
    )
    st.caption(
        "This unit comparison predicts archived positive reports—not "
        "true wildfire occurrence, satellite detection sensitivity, or "
        "complete observation opportunity. Changing the measurement "
        "unit does not make the satellite series equivalent."
    )
