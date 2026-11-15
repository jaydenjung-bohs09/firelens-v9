"""FireLens V10: Amazon research panel backed by verified NASA FIRMS archive summaries.

Does not alter or retrain the California ConvLSTM. No operational fire alerts.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from critical_periods import seasonal_screen

DATA = Path(__file__).resolve().parent / "processed" / "amazon"
TEAL = "#39D8C2"
ORANGE = "#FF9E54"
BLUE = "#77ACF8"
RED = "#FA6C75"

@st.cache_data(show_spinner=False)
def load_amazon_data():
    daily = pd.read_csv(DATA / "daily_summary.csv", parse_dates=["date_utc"])
    annual = pd.read_csv(DATA / "annual_summary.csv")
    spatial = pd.read_csv(DATA / "annual_spatial_1deg.csv")
    inventory = json.loads((DATA / "source_inventory.json").read_text())
    if len(daily) != 1461 or sorted(daily["year"].unique().tolist()) != [2022,2023,2024,2025]:
        raise ValueError("Amazon daily dates do not match NASA archive scope")
    if daily["union_positive_cells"].sum() != inventory["counts"]["union_positive_10km_cell_days"]:
        raise ValueError("Amazon daily cell totals failed source inventory check")
    if daily[["date_utc"]].duplicated().any().iloc[0]:
        raise ValueError("Duplicate Amazon UTC dates")
    if annual["union_positive_cell_days"].sum() != spatial["union_cell_days"].sum():
        raise ValueError("Spatial grid totals failed annual audit")
    return daily, annual, spatial, inventory

@st.cache_data(show_spinner=False)
def amazon_critical_screen(min_streak, ratio, percentile):
    daily, _, _, _ = load_amazon_data()
    dates = daily["date_utc"]
    # NASA-documented VIIRS S-NPP geolocation/command anomaly intervals;
    # conservative platform-level flags, not proof of no observation for a cell.
    july_flag = (
        dates.between("2024-07-09", "2024-07-16") |
        dates.between("2024-07-24", "2024-07-29")
    )
    m = pd.DataFrame({
        "target_date_utc": dates.dt.strftime("%Y-%m-%d"),
        "target_positive_cells": daily["union_positive_cells"],
        "target_modis_has_any_hotspot_record": daily["modis_has_positive_records"],
        "target_viirs_has_any_hotspot_record": daily["viirs_has_positive_records"],
        "target_snpp_known_anomaly_window": july_flag,
    })
    return seasonal_screen(
        m, window_days=14, peer_quantile=float(percentile),
        ratio_threshold=float(ratio), min_occupied_cells=20,
        min_peers=30, min_streak=int(min_streak)
    )

def _layout(fig, height=350, **kwargs):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#081322", plot_bgcolor="#102133",
        font=dict(color="#E8F0F6", size=12),
        height=height, margin=dict(l=20, r=15, t=54, b=25),
        legend=dict(orientation="h", y=-0.26),
        **kwargs
    )
    return fig

def render_amazon():
    st.caption("FIRELENS V10 RESEARCH PREVIEW / NASA FIRMS 2022–2025")
    st.header("Amazon Study Region · Four Years of Satellite Evidence")
    st.write(
        "Explore actual NASA FIRMS hotspot records within the selected "
        "Amazon study rectangle (74°W–48°W, 15°S–2°N). This rectangle "
        "crosses national borders; it is not the official Amazon boundary. "
        "Only NASA Type-0 **presumed vegetation-fire** records enter the "
        "visualizations. Presumed vegetation fire is not verified wildfire."
    )
    daily, annual, spatial, inv = load_amazon_data()
    total = inv["counts"]
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Raw archived records", f'{total["raw_total"]:,}')
    k2.metric("Type-0 records analyzed", f'{total["type0_total"]:,}')
    k3.metric("Occupied 10-km cell-days", f'{total["union_positive_10km_cell_days"]:,}')
    k4.metric("Days in archive window", f'{total["utc_days"]:,}')
    st.caption(
        "Four-year record totals combine sensors with unequal spatial resolution and "
        "sampling; they must not be interpreted as burned area, separate wildfire "
        "counts, or relative sensor accuracy."
    )

    yr = st.selectbox("Inspect calendar year", [2022,2023,2024,2025], index=2,
                      key="amazon_year")
    yday = daily.loc[daily.year.eq(yr)].copy()
    yspatial = spatial.loc[spatial.year.eq(yr)].copy()

    st.subheader("Four-year overview")
    annual_fig=go.Figure()
    annual_fig.add_trace(go.Bar(x=annual.year, y=annual.modis_positive_cell_days,
                                name="MODIS positive 10-km cell-days",
                                marker_color=BLUE))
    annual_fig.add_trace(go.Bar(x=annual.year, y=annual.viirs_positive_cell_days,
                                name="VIIRS S-NPP positive 10-km cell-days",
                                marker_color=ORANGE))
    annual_fig.add_trace(go.Scatter(x=annual.year, y=annual.union_positive_cell_days,
                                    name="Union positive 10-km cell-days",
                                    mode="lines+markers",line=dict(color=TEAL,width=3)))
    _layout(annual_fig, 355, title="Recorded cell-days by archive year",
            barmode="group", xaxis=dict(tickmode="array", tickvals=[2022,2023,2024,2025]),
            yaxis_title="Positive cell-days")
    st.plotly_chart(annual_fig, use_container_width=True)
    st.caption("Union counts one 10-km cell/UTC date once, even if both satellites record positives there. This is not a calibrated yearly wildfire trend.")

    monthly = (daily.assign(month=daily.date_utc.dt.month)
               .groupby(["year","month"]).union_positive_cells.sum()
               .unstack("month").reindex(index=[2022,2023,2024,2025],
                                        columns=range(1,13),fill_value=0))
    heat = go.Figure(go.Heatmap(
        z=monthly.to_numpy(),x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        y=["2022","2023","2024","2025"],
        colorscale=[[0,"#101F2B"],[.4,"#236576"],[.7,"#C98A43"],[1,"#FA7355"]],
        colorbar=dict(title="Cell-days"),
        hovertemplate="%{y} · %{x}<br>%{z:,.0f} union positive cell-days<extra></extra>"
    ))
    _layout(heat,270,title="Seasonality: monthly union positive 10-km cell-days",
            yaxis=dict(autorange="reversed"))
    st.plotly_chart(heat,use_container_width=True)

    st.subheader(f"{yr} daily recorded activity and evidence gaps")
    timeline=go.Figure()
    timeline.add_trace(go.Scatter(x=yday.date_utc,y=yday.union_positive_cells,
                                  name="Union positive cells",mode="lines",
                                  line=dict(color=TEAL,width=2)))
    timeline.add_trace(go.Scatter(x=yday.date_utc,y=yday.modis_positive_cells,
                                  name="MODIS positive cells",mode="lines",
                                  line=dict(color=BLUE,width=1,dash="dot")))
    timeline.add_trace(go.Scatter(x=yday.date_utc,y=yday.viirs_positive_cells,
                                  name="VIIRS positive cells",mode="lines",
                                  line=dict(color=ORANGE,width=1,dash="dot")))
    limited = yday.loc[~(yday.modis_has_positive_records & yday.viirs_has_positive_records)]
    if not limited.empty:
        timeline.add_trace(go.Scatter(
            x=limited.date_utc, y=limited.union_positive_cells,mode="markers",
            name="One/both sensors have no positive records",
            marker=dict(color=RED,size=8,symbol="x")))
    _layout(timeline,340,title="Daily union and sensor-specific 10-km cells",
            xaxis_title="UTC date",yaxis_title="Positive cells")
    st.plotly_chart(timeline,use_container_width=True)
    zdays=yday.loc[~yday.viirs_has_positive_records]
    if not zdays.empty:
        st.warning(
            f"{len(zdays)} UTC day(s) in {yr} contain no VIIRS S-NPP Type-0 "
            "positive records in this archive and region. These are **not "
            "confirmed fire-free days or independently verified complete sensor outages**."
        )
        with st.expander("See zero-VIIRS-record dates"):
            st.write(", ".join(zdays.date_utc.dt.strftime("%Y-%m-%d").tolist()))
    if yr==2024:
        st.info(
            "NASA has documented S-NPP platform anomalies in July 2024 "
            "(approximately July 9–16 and July 24–29). Interpret zero-record "
            "dates and cross-sensor differences cautiously; the broad anomaly "
            "windows do not provide per-grid-cell observation-coverage masks."
        )

    st.subheader(f"{yr} spatial pattern of positive cell-days")
    var = st.selectbox(
        "Map metric",
        ["Union positive 10-km cell-days", "MODIS positive 10-km cell-days",
         "VIIRS S-NPP positive 10-km cell-days"],
        key="amazon_map_metric"
    )
    col={"Union positive 10-km cell-days":"union_cell_days",
         "MODIS positive 10-km cell-days":"modis_cell_days",
         "VIIRS S-NPP positive 10-km cell-days":"viirs_cell_days"}[var]
    grid=(yspatial.pivot(index="lat_bin",columns="lon_bin",values=col)
          .reindex(index=range(-16,3), columns=range(-74,-47))
          .fillna(0))
    spatial_fig=go.Figure(go.Heatmap(
        x=np.array(grid.columns)+.5,y=np.array(grid.index)+.5,
        z=grid.to_numpy(), colorscale=[[0,"#101F2B"],[.35,"#236576"],[.75,"#C98A43"],[1,"#FA7355"]],
        colorbar=dict(title="Cell-days"),
        hovertemplate="Approx. bin center %{x:.1f}° lon, %{y:.1f}° lat<br>%{z:,.0f} cell-days<extra></extra>"
    ))
    _layout(spatial_fig,450,title=f"{yr} · Coarse 1° geographic summary",
            xaxis_title="Longitude (°)",yaxis_title="Latitude (°)",
            xaxis=dict(range=[-74.3,-47.8]),
            yaxis=dict(range=[-16.4,2.8],scaleanchor="x",scaleratio=1))
    st.plotly_chart(spatial_fig,use_container_width=True)
    st.caption(
        "The map groups projected 10-km equal-area cell centers into approximate "
        "1° latitude–longitude bins. These bins are not equal-area polygons or "
        "burned-area footprints; spatial comparisons require care."
    )

    st.divider()
    st.subheader("Regional critical-period screening")
    st.write(
        "Retrospective exploration: flag unusually high recorded activity "
        "relative to ±14 calendar days in **other** archive years. The screening "
        "requires positive records from both sensors and excludes broad documented "
        "July 2024 S-NPP anomaly windows. This is neither a fire alert nor a "
        "statistically calibrated anomaly detector."
    )
    c1,c2,c3=st.columns(3)
    with c1: run=st.slider("Consecutive candidate days",1,7,2,key="amazon_run")
    with c2: ratio=st.slider("Minimum relative to peer median",1.0,3.0,1.25,.05,key="amazon_ratio")
    with c3: percentile=st.select_slider("Peer percentile",options=[.75,.80,.85,.90,.95],value=.90,key="amazon_peer")
    crit, periods = amazon_critical_screen(run,ratio,percentile)
    yearcrit=crit.loc[crit.year.eq(yr)]
    filtered=periods.loc[periods.start_date.dt.year.eq(yr)].copy() if not periods.empty else periods.copy()
    d1,d2=st.columns(2)
    d1.metric("Candidate periods in selected year",len(filtered))
    d2.metric("Candidate individual days",int(yearcrit.candidate_day.sum()))
    if not filtered.empty:
        st.dataframe(filtered[["start_date","end_date","days","peak_occupied_cells",
                                "sum_excess_cell_days","peak_ratio"]],
                     use_container_width=True,hide_index=True)
    else:
        st.info("No periods match these screening settings in the selected year.")
    with st.expander("Data audit, methods and downloads"):
        st.json(inv)
        st.download_button("Download Amazon daily summary CSV",daily.to_csv(index=False),
                           "firelens_v10_amazon_daily.csv","text/csv")
        st.download_button("Download Amazon annual summary CSV",annual.to_csv(index=False),
                           "firelens_v10_amazon_annual.csv","text/csv")
        st.download_button("Download regional candidate periods CSV",periods.to_csv(index=False),
                           "firelens_v10_amazon_critical_periods.csv","text/csv")
    st.caption(
        "NASA FIRMS historical archive sources: MODIS Collection 6.1 and "
        "VIIRS S-NPP 375 m. Original files: fire_archive_M-C61_850616.csv "
        "and fire_archive_SV-C2_850617.csv. All results are positive-record "
        "proxies, not direct counts of unique fires or predictive accuracy. "
        "California ConvLSTM has not been retrained or independently validated "
        "on this Amazon region."
    )
