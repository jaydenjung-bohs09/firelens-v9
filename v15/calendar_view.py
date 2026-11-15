"""Country calendar UI: observed NOAA-20 versus modeled historical archive counts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from v15.calibration import country_month_rows
from v15.calendar import country_calendar,monthly_anomalies,selected_month_summary
from v15.validation import VALIDATION_CSV
from v15.validation_view import _out_of_sample_points

MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

@st.cache_data(show_spinner=False)
def _country_rows():
    return country_month_rows()

@st.cache_data(show_spinner=False)
def _validated_summary():
    return pd.read_csv(VALIDATION_CSV)

@st.cache_data(show_spinner=False)
def _country_calendar_cached(code):
    return country_calendar(_country_rows(),_validated_summary(),
                            _out_of_sample_points(),code)

def render_country_calendar():
    st.subheader("Country burning-activity record calendar · 2000–2025")
    st.caption(
        "Historical MODIS records mapped statistically onto a **VIIRS NOAA-20 "
        "positive-record count scale**. Before 2018 the values are estimates; "
        "from April 2018 they are actual NOAA-20 archive counts. These are "
        "not wildfire counts, burned area or physical satellite calibration."
    )
    data=_country_rows()
    names=(data.sort_values("year")
           .drop_duplicates("iso3").set_index("iso3")["country"].to_dict())
    codes=sorted([code for code in names if data.loc[data.iso3.eq(code),"modis"].notna().any()])
    default=codes.index("BRA") if "BRA" in codes else 0
    code=st.selectbox("Country or territory",codes,index=default,
                      format_func=lambda iso:f"{names.get(iso,iso)} ({iso})",
                      key="v15_calendar_country")
    table=_country_calendar_cached(code)
    view=st.radio("Calendar view",["Positive-record count","Difference from monthly median (%)"],
                  horizontal=True,key="v15_calendar_mode")
    rows=monthly_anomalies(table)
    value_col="value" if view=="Positive-record count" else "anomaly_pct"
    matrix=rows.pivot(index="year",columns="month",values=value_col).reindex(
        index=list(range(2000,2026)),columns=list(range(1,13)))
    custom=rows.pivot(index="year",columns="month",values="source").reindex(
        index=list(range(2000,2026)),columns=list(range(1,13)))
    custom_array=np.array(custom.fillna("missing").to_numpy(),dtype=object)
    z=matrix.to_numpy(dtype=float)
    if view=="Positive-record count":
        colors="Cividis"
        extra=dict(zmin=0)
        suffix=" records"
        hover_value="%{z:,.0f}"
    else:
        colors=[[0,"#2166AC"],[.5,"#F7F7F7"],[1,"#B35806"]]
        finite=np.abs(z[np.isfinite(z)])
        limit=max(25.0,float(np.quantile(finite,.90))) if len(finite) else 25.0
        extra=dict(zmin=-limit,zmax=limit,zmid=0)
        suffix=" %"
        hover_value="%{z:+.1f}"
    fig=go.Figure(go.Heatmap(
        x=list(range(1,13)),y=list(range(2000,2026)),
        z=z,customdata=custom_array,colorscale=colors,
        colorbar=dict(title=suffix.strip(),thickness=13),
        hovertemplate="%{y} · month %{x}<br>"+hover_value+suffix+
                      "<br>%{customdata} source<extra></extra>",
        **extra,
    ))
    fig.add_hline(y=2017.5,line_color="#FDF8E8",line_width=2,line_dash="dash",
                  annotation_text="2018: observed NOAA-20 begins in April",
                  annotation_position="top left")
    fig.update_layout(
        template="plotly_dark",height=610,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25,39,48,.75)",
        font=dict(color="#E6F0F7",size=12),
        margin=dict(l=28,r=24,t=50,b=42),
        xaxis=dict(tickmode="array",tickvals=list(range(1,13)),
                   ticktext=MONTHS,title="UTC acquisition month"),
        yaxis=dict(autorange="reversed",dtick=2,title="Year"),
    )
    st.plotly_chart(fig,use_container_width=True)
    st.caption(
        "2000–2017: modeled NOAA-20-equivalent archived positives (not satellite "
        "observations). 2018–2025: observed NOAA-20 archive counts, starting "
        "April 2018. Blank months have no supported estimate. The first MODIS "
        "source year starts November 2000. Anomaly colors saturate at the "
        "90th percentile of absolute anomalies for legibility; hover shows "
        "the underlying unmodified value."
        if view!="Positive-record count" else
        "2000–2017 is MODIS-modeled; 2018–2025 is observed NOAA-20 "
        "archive data (from April 2018). Blank indicates unavailable or "
        "undefined; the horizontal line separates modeled from observed years."
    )
    ycol,mcol=st.columns(2)
    with ycol:
        year=st.selectbox("Inspect year",list(range(2025,1999,-1)),
                          key="v15_calendar_selected_year")
    with mcol:
        month=st.selectbox("Inspect month",list(range(1,13)),
                           format_func=lambda m:MONTHS[m-1],
                           index=6,key="v15_calendar_selected_month")
    result=selected_month_summary(table,year,month)
    value=result["value"];median=result["median"];anomaly=result["anomaly_pct"]
    a,b,c,d=st.columns(4)
    a.metric("Selected month",f"{value:,.0f}" if np.isfinite(value) else "N/A",
             help="Source monthly positive-record count or modeled equivalent.")
    b.metric("Historical median",f"{median:,.0f}" if np.isfinite(median) else "N/A",
             help="Median of available same-calendar-month values across source years.")
    c.metric("Difference from median",f"{anomaly:+.1f}%" if np.isfinite(anomaly) else "N/A")
    d.metric("Rank",f"{result['rank']} of {result['comparable_years']}"
             if result["rank"] else "N/A")
    st.caption(
        "For modeled months, lower/upper bounds form a nominal 80% interval "
        "based on 2019–2025 held-out log-residual quantiles, not verified "
        "historical prediction coverage. Missing MODIS-zero months remain undefined."
    )
    st.download_button(
        "Download selected country's monthly calendar (CSV)",
        table.to_csv(index=False,columns=[
            "country","year","month","value","source","lower","upper"
        ]),
        file_name=f"MODVIIR_{code}_2000_2025_Calendar.csv",
        mime="text/csv",key="v15_calendar_download"
    )
