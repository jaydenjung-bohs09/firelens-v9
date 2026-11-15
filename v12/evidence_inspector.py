"""California Park Fire Evidence Inspector: record-level, time-aware NASA FIRMS audit.

Only *positive archive records* are present. Cell, date and phase matching
does not establish simultaneous coverage, independent satellite negatives,
individual fire identity or operational forecasting accuracy.

Underlying archive provenance and geometric filtering were established in
the preserved V8 workflow; this module does not change those inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer

GRID_M = 5000
FORWARD = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
REVERSE = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)

EVIDENCE_ORDER = [
    "Both positive / time-compatible",
    "Both positive / time-separated",
    "MODIS positive / S-NPP flagged",
    "MODIS positive / VIIRS opportunity unknown",
    "VIIRS positive / MODIS opportunity unknown",
]
EVIDENCE_COLORS = {
    EVIDENCE_ORDER[0]: "#39D8C2",
    EVIDENCE_ORDER[1]: "#F6AE60",
    EVIDENCE_ORDER[2]: "#FA6C75",
    EVIDENCE_ORDER[3]: "#77ACF8",
    EVIDENCE_ORDER[4]: "#B39CE6",
}

def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1"}


def match_acquisition_times(modis, viirs, tolerance_minutes: float):
    """Maximum-cardinality, one-to-one match of sorted minute-of-day values.

    Each tuple is (MODIS minute, VIIRS minute, unsigned time difference).
    This uses the frozen V8 chronological two-pointer algorithm; it does not
    assert physical identity of detections or per-pixel observation coverage.
    """
    if tolerance_minutes < 0:
        raise ValueError("Time tolerance must be nonnegative")
    a = np.sort(np.asarray(modis, dtype=float))
    b = np.sort(np.asarray(viirs, dtype=float))
    i = j = 0
    pairs = []
    while i < len(a) and j < len(b):
        delta = float(a[i] - b[j])
        if abs(delta) <= tolerance_minutes:
            pairs.append((float(a[i]), float(b[j]), abs(delta)))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return pairs


@st.cache_data(show_spinner=False)
def prepare_park_records(records: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "latitude", "longitude", "sensor_group",
                "daynight", "utc_minute", "satellite", "frp", "confidence", "type"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Park record archive missing: {sorted(missing)}")
    r = records.copy()
    r["date"] = pd.to_datetime(r["date"], errors="raise").dt.strftime("%Y-%m-%d")
    r["utc_minute"] = pd.to_numeric(r["utc_minute"], errors="coerce")
    r["latitude"] = pd.to_numeric(r["latitude"], errors="coerce")
    r["longitude"] = pd.to_numeric(r["longitude"], errors="coerce")
    r["type"] = pd.to_numeric(r["type"], errors="coerce")
    ok = (r["sensor_group"].isin(["MODIS", "VIIRS"])
          & r["daynight"].isin(["D", "N"])
          & r["latitude"].between(-90, 90)
          & r["longitude"].between(-180, 180)
          & r["utc_minute"].between(0, 1439))
    if not ok.all():
        raise ValueError(f"Invalid Park records: {int((~ok).sum())}")
    x, y = FORWARD.transform(r.longitude.to_numpy(), r.latitude.to_numpy())
    r["grid_x"] = np.floor(np.asarray(x) / GRID_M).astype("int64")
    r["grid_y"] = np.floor(np.asarray(y) / GRID_M).astype("int64")
    r["cell_id"] = r.grid_x.astype(str) + "_" + r.grid_y.astype(str)
    r["acquisition_utc"] = r["utc_minute"].map(
        lambda value: f"{int(value)//60:02d}:{int(value)%60:02d}"
    )
    return r.sort_values(["date", "cell_id", "daynight",
                          "sensor_group", "utc_minute"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def rebuild_evidence_groups(records: pd.DataFrame, sensor_status: pd.DataFrame,
                            tolerance_minutes: int = 180,
                            type0_only: bool = False) -> pd.DataFrame:
    """One row for each positive-report cell / UTC date / D-N phase."""
    r = prepare_park_records(records)
    if type0_only:
        r = r.loc[r["type"].eq(0)].copy()
    if r.empty:
        raise ValueError("No positive Park Fire records after filtering")
    status = sensor_status.copy()
    status["UTC date"] = pd.to_datetime(
        status["UTC date"], errors="raise").dt.strftime("%Y-%m-%d")
    context = {
        row["UTC date"]: (
            _as_bool(row["S-NPP anomaly window"]),
            _as_bool(row["FIRMS listed VIIRS missing date"]),
        )
        for _, row in status.iterrows()
    }
    rows = []
    for (date, cell_id, phase), group in r.groupby(
            ["date", "cell_id", "daynight"], sort=True):
        modis = group.loc[group.sensor_group.eq("MODIS"), "utc_minute"].to_numpy()
        viirs = group.loc[group.sensor_group.eq("VIIRS"), "utc_minute"].to_numpy()
        pairings = match_acquisition_times(modis, viirs, tolerance_minutes)
        anomaly, missing_date = context.get(date, (False, False))
        both = bool(len(modis) and len(viirs))
        if both:
            category = EVIDENCE_ORDER[0] if pairings else EVIDENCE_ORDER[1]
        elif len(modis):
            category = (EVIDENCE_ORDER[2] if anomaly or missing_date
                        else EVIDENCE_ORDER[3])
        else:
            category = EVIDENCE_ORDER[4]
        gx = int(group["grid_x"].iloc[0])
        gy = int(group["grid_y"].iloc[0])
        center_lon, center_lat = REVERSE.transform(
            (gx + .5) * GRID_M, (gy + .5) * GRID_M
        )
        if both:
            # Minimum potential gap is independent of one-to-one pairing and
            # supports a transparent tolerance-sensitivity curve.
            nearest_gap = float(np.min(np.abs(
                modis[:, None] - viirs[None, :])))
        else:
            nearest_gap = np.nan
        rows.append({
            "UTC date": date,
            "Cell ID": cell_id,
            "Day/night": phase,
            "MODIS records": int(len(modis)),
            "VIIRS records": int(len(viirs)),
            "Both sensors positive": both,
            "Time-compatible": bool(pairings),
            "One-to-one matched pairs": len(pairings),
            "Closest acquisition gap (min)": nearest_gap,
            "Median matched-pair gap (min)": (
                float(np.median([p[2] for p in pairings]))
                if pairings else np.nan
            ),
            "S-NPP anomaly window": anomaly,
            "FIRMS listed VIIRS missing date": missing_date,
            "Evidence class": category,
            "Observation coverage verified": False,
            "cell_center_lon": float(center_lon),
            "cell_center_lat": float(center_lat),
        })
    return pd.DataFrame(rows)


def _fig_style(fig, height=350, **kwargs):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,25,38,.72)", font_color="#E8F0F6",
        margin=dict(l=30, r=20, t=54, b=45),
        hoverlabel=dict(bgcolor="#0F2133", font_size=12),
        **kwargs,
    )
    return fig


def _plot_perimeter(fig, geometry):
    if not isinstance(geometry, dict):
        return
    coords = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        polygons = [coords]
    elif geometry.get("type") == "MultiPolygon":
        polygons = coords
    else:
        return
    for polygon in polygons:
        if not polygon:
            continue
        ring = polygon[0]
        if not ring:
            continue
        step = max(1, len(ring) // 500)
        points = ring[::step]
        if points[-1] != ring[-1]:
            points.append(ring[-1])
        fig.add_trace(go.Scattergl(
            x=[v[0] for v in points], y=[v[1] for v in points],
            mode="lines", line=dict(color="#B6A47C", width=1.1),
            hoverinfo="skip", showlegend=False, name="Final historical footprint",
        ))


def _clock(minutes):
    m = int(round(float(minutes)))
    return f"{m//60:02d}:{m%60:02d}"


def render_inspector(records: pd.DataFrame, status: pd.DataFrame,
                     archived_groups: pd.DataFrame, geometry):
    st.markdown("### Satellite Evidence Inspector")
    st.caption("PARK FIRE · JULY 2024 · INTERACTIVE POSITIVE-RECORD AUDIT")
    st.write(
        "Inspect a **5-km equal-area cell, UTC date, and day/night phase**. "
        "Change the acquisition-time tolerance to see which MODIS and "
        "VIIRS S-NPP positives are temporally compatible. This is not a "
        "measure of fire detection accuracy: no independent satellite "
        "observation-opportunity masks or verified negative cells are available."
    )
    ctl1, ctl2, ctl3 = st.columns([1.5, 1, 1])
    with ctl1:
        tolerance = st.slider(
            "Maximum acquisition-time gap (minutes)",
            min_value=0, max_value=720, value=180, step=30,
            key="v12_time_tolerance",
            help="A matched pair requires the same UTC date, day/night phase, "
                 "5-km projected grid cell, and acquisition times within this tolerance."
        )
    with ctl2:
        type0_only = st.checkbox(
            "Presumed vegetation fires (type 0) only",
            value=False, key="v12_type0_only",
            help="Off reproduces all 3,092 archived Park records, including "
                 "eight type-2 thermal anomalies; changing it changes the denominators."
        )
    with ctl3:
        exclude_flags = st.checkbox(
            "Exclude flagged S-NPP dates from rate",
            value=False, key="v12_exclude_flags",
            help="Flagged dates remain accessible in the evidence inspector; "
                 "this checkbox only changes the summary denominator."
        )

    groups = rebuild_evidence_groups(records, status, tolerance, type0_only)
    base = groups
    if exclude_flags:
        base = groups.loc[
            ~(groups["S-NPP anomaly window"]
              | groups["FIRMS listed VIIRS missing date"])
        ].copy()
    both = int(base["Both sensors positive"].sum())
    match = int(base["Time-compatible"].sum())
    active_groups = len(base)
    a, b, c, d = st.columns(4)
    a.metric("Positive-report groups", f"{active_groups:,}")
    b.metric("Both sensors positive", f"{both:,}",
             f"{100*both/active_groups:.1f}% of groups"
             if active_groups else "No eligible groups", delta_color="off")
    c.metric("Within selected tolerance", f"{match:,}",
             f"{100*match/both:.1f}% of both-positive"
             if both else "No both-positive groups", delta_color="off")
    d.metric("One-to-one time pairs", f"{int(base['One-to-one matched pairs'].sum()):,}")

    if tolerance == 180 and not type0_only and not exclude_flags:
        expected = (316, 54, 45)
        observed = (active_groups, both, match)
        if observed != expected:
            st.error(
                f"Archived reference mismatch: expected {expected}, got {observed}. "
                "Do not cite the audit until the input records are reconciled."
            )
            st.stop()
        st.caption(
            "Integrity check: exactly reproduces the archived three-hour "
            "316 / 54 / 45 group audit. The group universe includes at least "
            "one reported positive in either sensor; it excludes unknown "
            "no-record cells."
        )
    if exclude_flags:
        st.warning(
            "The headline denominator excludes broad S-NPP anomaly dates. "
            "This selection does not verify complete coverage on remaining days."
        )

    st.divider()
    st.markdown("#### Inspect a date and an individual grid cell")
    dates = sorted(groups["UTC date"].unique().tolist())
    cd, cp = st.columns([1.2, 1])
    with cd:
        selected_date = st.selectbox(
            "UTC date with positive records", dates, index=len(dates) - 2,
            key="v12_date")
    with cp:
        phase = st.radio("Day/night phase", ["Both", "Day", "Night"],
                         horizontal=True, key="v12_phase")
    day = groups.loc[groups["UTC date"].eq(selected_date)].copy()
    if phase != "Both":
        day = day.loc[day["Day/night"].eq("D" if phase == "Day" else "N")].copy()

    # The inspector deliberately includes flagged dates even when the
    # "exclude from rate" option is active.
    flagged = day["S-NPP anomaly window"].any() or day[
        "FIRMS listed VIIRS missing date"].any()
    if flagged:
        st.warning(
            f"{selected_date}: a documented S-NPP platform anomaly or FIRMS "
            "missing-data date affects interpretation. This is not a per-cell "
            "coverage determination."
        )
    else:
        st.info(
            "No known S-NPP anomaly flag for this date in the included source "
            "metadata. Satellite coverage of each unreported cell remains unverified."
        )
    if day.empty:
        st.info("No positive records in this archive subset for the chosen phase.")
        return

    # Date-specific cell/phase selection is stable across chart redraws.
    keys = list(day.sort_values(
        ["MODIS records", "VIIRS records"], ascending=False
    ).index)
    selected_index = st.selectbox(
        "Inspect grid cell (5 km)", keys,
        format_func=lambda ix: (
            f"{day.loc[ix, 'Cell ID']} · "
            f"{'Day' if day.loc[ix, 'Day/night']=='D' else 'Night'} · "
            f"M {int(day.loc[ix, 'MODIS records'])} / "
            f"V {int(day.loc[ix, 'VIIRS records'])} · "
            f"{day.loc[ix, 'Evidence class']}"
        ), key="v12_selected_group"
    )
    chosen = day.loc[selected_index]
    mapcol, detailcol = st.columns([1.55, 1.0])
    with mapcol:
        fig = go.Figure()
        _plot_perimeter(fig, geometry)
        for category in EVIDENCE_ORDER:
            subset = day.loc[day["Evidence class"].eq(category)]
            if subset.empty:
                continue
            custom = np.stack([
                subset["Cell ID"].astype(str).to_numpy(),
                subset["Day/night"].astype(str).to_numpy(),
                subset["MODIS records"].astype(int).to_numpy(),
                subset["VIIRS records"].astype(int).to_numpy(),
            ], axis=1)
            fig.add_trace(go.Scattergl(
                x=subset.cell_center_lon, y=subset.cell_center_lat,
                mode="markers",
                name=category,
                marker=dict(size=8.5, color=EVIDENCE_COLORS[category],
                            opacity=.9, line=dict(color="#112638", width=.5)),
                customdata=custom,
                hovertemplate=(
                    "Cell %{customdata[0]} · phase %{customdata[1]}"
                    "<br>MODIS %{customdata[2]} / VIIRS %{customdata[3]}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ))
        fig.add_trace(go.Scattergl(
            x=[chosen.cell_center_lon], y=[chosen.cell_center_lat],
            mode="markers", name="Selected cell",
            marker=dict(size=18, color="rgba(0,0,0,0)",
                        line=dict(color="white", width=2.6)),
            hoverinfo="skip"
        ))
        _fig_style(
            fig, 435, title=f"{selected_date} · Positive-record cell/phase map",
            xaxis_title="Longitude", yaxis_title="Latitude",
            legend=dict(orientation="h", y=-.23, font_size=9)
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Markers show the projected grid-cell center. The outline is "
            "the retrospective final Park Fire footprint—not its daily extent. "
            "Some boundary-cell centers may lie outside the polygon."
        )
    with detailcol:
        st.markdown("#### Evidence interpretation")
        st.markdown(f"**{chosen['Evidence class']}**")
        st.write(
            f"**{chosen['Cell ID']}** · "
            f"{'Day' if chosen['Day/night']=='D' else 'Night'}"
        )
        st.write(f"MODIS positive records: **{int(chosen['MODIS records'])}**")
        st.write(f"VIIRS S-NPP positive records: **{int(chosen['VIIRS records'])}**")
        if chosen["Both sensors positive"]:
            st.write(
                f"Closest observed acquisition-time gap: "
                f"**{chosen['Closest acquisition gap (min)']:.0f} min**"
            )
            st.write(
                f"One-to-one pairs within {tolerance} min: "
                f"**{int(chosen['One-to-one matched pairs'])}**"
            )
        else:
            st.write("Closest cross-sensor acquisition gap: **not defined**")
        if chosen["Both sensors positive"] and chosen["Time-compatible"]:
            st.success(
                "Both sensors reported at least one positive in this cell, "
                "date and phase within the selected time tolerance."
            )
        elif chosen["Both sensors positive"]:
            st.info(
                "Both sensors reported a positive here on this date and "
                "phase, but none within the selected acquisition-time tolerance."
            )
        else:
            st.info(
                "Only one sensor has positive reports in this cell/date/phase "
                "group. The other sensor's observation opportunity is unknown."
            )
        st.caption(
            "**Not established:** no-fire observations, exhaustive satellite "
            "coverage, common fire identity, detection sensitivity, burned "
            "area or incident progression."
        )

    filtered_records = prepare_park_records(records)
    if type0_only:
        filtered_records = filtered_records.loc[filtered_records["type"].eq(0)]
    selected_records = filtered_records.loc[
        filtered_records["date"].eq(selected_date)
        & filtered_records["cell_id"].eq(chosen["Cell ID"])
        & filtered_records["daynight"].eq(chosen["Day/night"])
    ].copy()

    st.markdown("#### Actual satellite acquisitions for the selected cell")
    sequence_fig = go.Figure()
    pairs = match_acquisition_times(
        selected_records.loc[selected_records.sensor_group.eq("MODIS"), "utc_minute"],
        selected_records.loc[selected_records.sensor_group.eq("VIIRS"), "utc_minute"],
        tolerance,
    )
    for ma, vi, gap in pairs:
        sequence_fig.add_trace(go.Scatter(
            x=[ma, vi], y=[1, 0], mode="lines",
            line=dict(color="#39D8C2", width=2, dash="dot"),
            hovertemplate=f"Pair gap {gap:.0f} minutes<extra></extra>",
            showlegend=False,
        ))
    for sensor, yy, color in [("MODIS", 1, "#77ACF8"),
                               ("VIIRS", 0, "#FF9E54")]:
        portion = selected_records.loc[selected_records.sensor_group.eq(sensor)]
        if portion.empty:
            continue
        sequence_fig.add_trace(go.Scatter(
            x=portion["utc_minute"], y=[yy] * len(portion),
            name=sensor, mode="markers",
            marker=dict(size=11, color=color,
                        line=dict(color="#111D29", width=1)),
            text=[f"UTC {v} · FRP {f:.1f} MW"
                  for v, f in zip(portion["acquisition_utc"],
                                  pd.to_numeric(portion["frp"], errors="coerce"))],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
        ))
    _fig_style(
        sequence_fig, 290,
        title=f"{selected_date} UTC acquisitions · dotted segments are one-to-one pairs",
        xaxis=dict(
            range=[0, 1440], title="Time in UTC",
            tickvals=[0, 240, 480, 720, 960, 1200, 1440],
            ticktext=["00:00","04:00","08:00","12:00","16:00","20:00","24:00"],
        ),
        yaxis=dict(
            tickvals=[0, 1], ticktext=["VIIRS S-NPP", "MODIS"],
            range=[-.5, 1.5], title=""
        ),
        legend=dict(orientation="h", y=-.35),
    )
    st.plotly_chart(sequence_fig, use_container_width=True)
    rows = selected_records[[
        "sensor_group", "satellite", "acquisition_utc", "latitude",
        "longitude", "frp", "confidence", "type"
    ]].rename(columns={
        "sensor_group": "Sensor", "satellite": "Satellite",
        "acquisition_utc": "Time (UTC)", "latitude": "Latitude",
        "longitude": "Longitude", "frp": "FRP (MW)",
        "confidence": "Source confidence", "type": "NASA type",
    })
    st.dataframe(rows, hide_index=True, use_container_width=True, height=260)
    x1, x2 = st.columns(2)
    with x1:
        st.download_button(
            "Export selected satellite records",
            selected_records.to_csv(index=False),
            file_name=f"firelens_park_{selected_date}_{chosen['Cell ID']}_{chosen['Day/night']}.csv",
            mime="text/csv", key="v12_export_cell"
        )
    with x2:
        st.download_button(
            "Export full recalculated audit",
            groups.to_csv(index=False),
            file_name=f"firelens_park_audit_{tolerance}min.csv",
            mime="text/csv", key="v12_export_audit"
        )

    st.divider()
    st.markdown("#### Does the matching tolerance change the conclusion?")
    joint = base.loc[base["Both sensors positive"]]
    tolerances = [0, 30, 60, 120, 180, 240, 360, 720]
    counts = [
        int((joint["Closest acquisition gap (min)"] <= v).sum())
        for v in tolerances
    ]
    sensitivity = go.Figure(go.Scatter(
        x=np.asarray(tolerances) / 60, y=counts, mode="lines+markers",
        line=dict(color="#39D8C2", width=3), marker=dict(size=8),
        text=[f"{k} of {len(joint)} both-positive groups" for k in counts],
        hovertemplate="%{x:.1f} h tolerance<br>%{text}<extra></extra>",
    ))
    sensitivity.add_vline(x=tolerance/60, line_dash="dot",
                          line_color="#F6AE60")
    _fig_style(
        sensitivity, 285, title="Sensitivity of any-pair temporal compatibility",
        xaxis_title="Maximum acquisition-time gap (hours)",
        yaxis_title="Both-positive groups with ≥1 compatible pair",
        yaxis_range=[0, max(10, len(joint) + 3)],
    )
    st.plotly_chart(sensitivity, use_container_width=True)
    st.caption(
        "The curve reports groups with **at least one potential acquisition-time "
        "pair**, not independent observed fire locations, physical matching "
        "certainty, or one-to-one pair counts. The raw V8 reference at 180 "
        "minutes is preserved and verifiable. Restricting to NASA type 0 "
        "removes non-vegetation thermal anomalies and alters the group universe."
    )
