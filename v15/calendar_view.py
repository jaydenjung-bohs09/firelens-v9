"""Country calendar UI: observed NOAA-20 versus modeled historical archive counts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from v15.calibration import country_month_rows,fit_calibration,complete_overlap_years
from v15.calendar import country_calendar,monthly_anomalies,selected_month_summary
from v15.validation import VALIDATION_CSV,selected_method,prediction_intervals
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


@st.cache_data(show_spinner=False, max_entries=8)
def _continuity_series(code):
    source=_country_rows()
    selected=source.loc[source.iso3.eq(code)&source.year.between(2001,2025)].copy()
    if selected.empty:
        return pd.DataFrame()
    method=selected_method(_validated_summary())
    fitted=fit_calibration(source,years=complete_overlap_years(source))
    modeled=fitted.predict(selected,method)
    qlo,qhi=prediction_intervals(_out_of_sample_points(),method).get(
        str(selected.continent.iloc[0]),prediction_intervals(_out_of_sample_points(),method)["GLOBAL"]
    )
    modeled["lower"]=np.maximum(0,np.expm1(np.log1p(modeled.predicted)+qlo))
    modeled["upper"]=np.expm1(np.log1p(modeled.predicted)+qhi)
    annual=[]
    for year,group in modeled.groupby("year"):
        full=(len(group)==12 and set(group.month)==set(range(1,13))
              and group.predicted.notna().all())
        observed_full=(year>=2019 and len(group)==12 and group.viirs.notna().all())
        annual.append(dict(
            year=int(year),
            modeled=float(group.predicted.sum()) if full else np.nan,
            lower=float(group.lower.sum()) if full else np.nan,
            upper=float(group.upper.sum()) if full else np.nan,
            observed=float(group.viirs.sum()) if observed_full else np.nan,
        ))
    return pd.DataFrame(annual)


def render_continuity_chart(code):
    st.subheader("Does the MODIS-based estimate track what VIIRS actually recorded?")
    annual=_continuity_series(code)
    if annual.empty or annual.modeled.notna().sum()<2:
        st.info("Insufficient complete-year, nonzero MODIS observations for this country.")
        return
    fig=go.Figure()
    valid=annual.loc[annual.modeled.notna()].sort_values("year")
    fig.add_trace(go.Scatter(x=valid.year,y=valid.upper,mode="lines",
                            line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=valid.year,y=valid.lower,mode="lines",
                            fill="tonexty",fillcolor="rgba(57,216,194,.16)",
                            line=dict(width=0),name="Nominal 80% band",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=valid.year,y=valid.modeled,mode="lines+markers",
                            name="MODIS-based modeled NOAA-20-equivalent",
                            line=dict(color="#39D8C2",width=2)))
    observed=annual.loc[annual.year.between(2019,2025)&annual.observed.notna()]
    if not observed.empty:
        fig.add_trace(go.Scatter(x=observed.year,y=observed.observed,
                                mode="lines+markers",name="Observed VIIRS NOAA-20",
                                line=dict(color="#FFAE65",width=2)))
    fig.update_layout(template="plotly_dark",height=360,
                      xaxis=dict(title="Full calendar year",dtick=2),
                      yaxis_title="Annual archived positive reports",
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,33,51,.8)",
                      margin=dict(l=30,r=20,t=105,b=45),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                  xanchor="left",x=0))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("2001–2002: Terra-only MODIS — modeled values likely low; "
               "Aqua began observations in 2002. The fitted calibration uses later Terra+Aqua years.")
    st.caption("Annual values count satellite positive-record reports, not individual fires; "
               "modeled intervals are empirical held-out residual bands, not verified historical coverage.")

def render_country_calendar():
    st.subheader("Country burning-activity record calendar · 2000–2025")
    st.caption(
        "Historical MODIS records mapped statistically onto a **VIIRS NOAA-20 "
        "positive-record count scale**. Before 2018 the values are estimates; "
        "from April 2018 they are actual NOAA-20 archive counts. Jan–Mar 2018 "
        "are MODIS-based modeled values because the NOAA-20 archive is absent. These are " 
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
    fig.add_hrect(y0=1999.5,y1=2002.5,fillcolor="#D8D7CF",opacity=.17,
                  line_width=0,layer="above",
                  annotation_text="Terra-only MODIS — modeled values likely low",
                  annotation_position="top right")
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
        "2000–2017 and Jan–Mar 2018: modeled NOAA-20-equivalent positives, "
        "not observed NOAA-20. April 2018 onward: observed NOAA-20 records. "
        "Blank months have no supported estimate. The first MODIS " 
        "source year starts November 2000. Anomaly colors saturate at the "
        "90th percentile of absolute anomalies for legibility; hover shows "
        "the underlying unmodified value."
        if view!="Positive-record count" else
        "2000–2017 and Jan–Mar 2018 are MODIS-modeled; April 2018 onward "
        "is observed NOAA-20 archive data. Blank indicates unavailable or " 
        "undefined; the horizontal line separates modeled from observed years."
    )
    st.caption("2000–2002: Terra-only MODIS — modeled values likely low; "
               "Aqua joined during 2002, so its early-era record coverage differs "
               "from the later Terra+Aqua calibration training period.")
    render_continuity_chart(code)
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
