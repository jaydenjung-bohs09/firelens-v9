"""MODVIIR — global map and country-level seasonal comparison.

NASA FIRMS archived active-fire/hotspot observations. Distinct sensors are
never implicitly treated as equivalent or converted to unique wildfire counts.
"""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import jensenshannon

BASE = Path(__file__).resolve().parent / "processed" / "global"
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MCOLS = [f"m{i:02d}" for i in range(1,13)]
SENSORS = {
    "MODIS": "MODIS · Terra/Aqua · 2000–2025",
    "VIIRS_NOAA20": "VIIRS · NOAA-20 · 2018–2025"
}
TEAL, BLUE, AMBER = "#39D8C2", "#77ACF8", "#FFAE65"
COLOR = ["#39D8C2","#FFAE65","#77ACF8","#F97584","#C9A0FC"]
SCALE = [[0,"#10202D"],[.2,"#195468"],[.45,"#258F91"],[.75,"#E9A14D"],[1,"#FF6D5A"]]
CAVEAT = ("NASA FIRMS type-0 observations are presumed vegetation-fire hotspot "
          "records—not distinct wildfires, burned area, fire probability, or "
          "a measure of sensor detection accuracy. Raw MODIS and VIIRS counts "
          "are not sensor-normalized and must not be compared as equivalent.")


@st.cache_data(show_spinner=False)
def load_world():
    df = pd.read_csv(BASE / "global_country_year.csv", dtype={"iso3":"string"})
    metadata = json.loads((BASE / "global_manifest.json").read_text())
    if len(df) != metadata["country_year_rows"]:
        raise ValueError("Global archive row-count integrity check failed")
    numeric = ["vegetation_records","raw_records","active_record_days"] + MCOLS
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    df["year"] = df["year"].astype(int)
    if int(df.vegetation_records.sum()) != metadata["total_vegetation_records"]:
        raise ValueError("Global archive count integrity check failed")
    if not np.array_equal(df[MCOLS].sum(axis=1).to_numpy(),df.vegetation_records.to_numpy()):
        raise ValueError("Monthly archive total integrity check failed")
    if df.duplicated(["sensor","year","iso3"]).any():
        raise ValueError("Duplicated country-year satellite inventory")
    return df, metadata


def headline(kicker, title, subtitle):
    st.markdown(f"<div style='font-size:.74rem;color:{TEAL};font-weight:800;letter-spacing:.15em'>{kicker.upper()}</div>",
                unsafe_allow_html=True)
    st.title(title)
    st.markdown(subtitle)


def style(fig, height=350, **settings):
    fig.update_layout(
        template="plotly_dark",height=height,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(16,34,48,.75)",
        font=dict(color="#E7EFF7",size=12),
        margin=dict(l=20,r=16,t=44,b=40),
        hoverlabel=dict(bgcolor="#102F42"),
        legend=dict(orientation="h",y=-.27),**settings
    )
    return fig


def selector(key):
    return st.radio(
        "Satellite product", list(SENSORS),
        format_func=lambda s:SENSORS[s], horizontal=True,key=key
    )


def evidence_notes():
    with st.expander("Data provenance, coverage and interpretation"):
        st.markdown("**NASA/LANCE FIRMS yearly country archives.** MODIS Collection "
                    "6.1 (2000–2025) and VIIRS NOAA-20/JPSS-1 (2018–2025). "
                    "The first MODIS year starts in November 2000, and NOAA-20 "
                    "in April 2018; treat these as partial years.")
        st.warning(CAVEAT)
        st.markdown("Country totals include agricultural and managed burning and "
                    "can reflect cloud cover, overpass opportunities, country size, "
                    "classification uncertainty and processing changes. Missing "
                    "positive records do not prove a satellite observed no fire. "
                    "This country-level archive has no valid pixel-level negative "
                    "observations or observation-opportunity normalization.")
        st.markdown("[NASA FIRMS download](https://firms.modaps.eosdis.nasa.gov/download/) · "
                    "[NASA challenge](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)")
        st.caption(
            "Future work: near-real-time FIRMS area-API coverage. "
            "This archived dashboard ends in 2025 and does not currently "
            "connect to the area API or store an API key."
        )


def render_global_atlas():
    df, meta = load_world()
    headline(
        "01 / Global Atlas · NASA FIRMS",
        "Earth's burning signals, across decades",
        "Explore **where and when NASA satellites recorded presumed vegetation-fire hotspots**. "
        "The default view displays the statistically calibrated NOAA-20 archive-count calendar. "
        "Use the raw-sensor switch below to inspect unadjusted MODIS and VIIRS records."
    )
    from v15.validation import VALIDATION_CSV
    from v15.calendar_view import render_country_calendar
    from v15.validation_view import render_validation
    calibration = pd.read_csv(VALIDATION_CSV)
    global_rows = calibration.loc[calibration.continent.eq("GLOBAL")].set_index("method")
    unadjusted = global_rows.loc["none"]
    calibrated = global_rows.loc["ratio"]
    a, b, c = st.columns(3)
    a.metric("Uncalibrated error",help="Held-out median absolute percentage error for unadjusted MODIS",
             value=f"{unadjusted.median_abs_pct_error:.1f}%")
    b.metric("Calibrated error",help="Held-out median absolute percentage error for monthly country ratio",
             value=f"{calibrated.median_abs_pct_error:.1f}%")
    c.metric("Tested on",
             f"{int(calibrated.n_predictions):,} country-months, "
             f"{str(calibrated.holdout_years).split(',')[0]}–"
             f"{str(calibrated.holdout_years).split(',')[-1]}")
    st.caption("Calibration was trained on other years and tested on each withheld year.")
    render_country_calendar()
    with st.expander("How we tested the calibration (held-out years)", expanded=False):
        render_validation()
    st.divider()
    show_raw = st.checkbox(
        "Show raw MODIS / VIIRS archive charts",
        value=False,key="v15_show_raw_sensor_atlas",
        help="The unadjusted records remain separate by sensor and have not been normalized for observation opportunity.",
    )
    if not show_raw:
        evidence_notes()
        return
    sensor = selector("v11_global_sensor")
    years = sorted(df.loc[df.sensor.eq(sensor),"year"].unique().tolist())
    year = st.select_slider("Explore archive year",options=years,
                            value=2024 if 2024 in years else years[-1],
                            key="v11_global_year")
    selected = df.loc[df.sensor.eq(sensor)&df.year.eq(year)&df.vegetation_records.gt(0)].copy()
    selected = selected.sort_values("vegetation_records",ascending=False)
    annual = (df.loc[df.sensor.eq(sensor)].groupby("year",as_index=False)
              [["vegetation_records"]+MCOLS].sum().sort_values("year"))
    a,b,c,d=st.columns(4)
    a.metric("NASA type-0 records",f"{int(selected.vegetation_records.sum()):,}")
    b.metric("Countries / areas",help="Countries and territories represented by source records",value=str(len(selected)))
    c.metric("Most archived records",selected.country.iloc[0] if len(selected) else "—")
    d.metric("Years represented",str(len(years)))
    if year==years[0]:
        st.info("This is the satellite product's **partial first archive year**, not a full 12-month observation year.")
    st.subheader(f"{year} · Worldwide positive-record distribution")
    left,right=st.columns([1.8,1],gap="large")
    with left:
        chart=go.Figure(go.Choropleth(
            locations=selected.iso3,locationmode="ISO-3",
            z=np.log10(1+selected.vegetation_records.to_numpy(dtype=float)),
            text=selected.country,customdata=selected.vegetation_records,
            colorscale=SCALE,zmin=0,
            marker_line_color="#314A57",marker_line_width=.35,
            colorbar=dict(title="log₁₀(count + 1)",thickness=12,len=.72),
            hovertemplate="<b>%{text}</b><br>%{customdata:,.0f} archived records<extra></extra>"
        ))
        chart.update_geos(projection_type="natural earth",
                          showframe=False,showcoastlines=True,
                          coastlinecolor="#375368",showcountries=True,
                          countrycolor="#3A5869",bgcolor="rgba(0,0,0,0)")
        style(chart,430,geo=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(chart,use_container_width=True)
    with right:
        ranking=selected.head(12).iloc[::-1]
        chart=go.Figure(go.Bar(
            x=ranking.vegetation_records,y=ranking.country,
            orientation="h",marker_color=AMBER,
            hovertemplate="%{y}<br>%{x:,.0f} positive records<extra></extra>"
        ))
        style(chart,430,xaxis_title="Type-0 archived records")
        st.plotly_chart(chart,use_container_width=True)
    st.caption("Color shows log-scaled NASA hotspot-record counts by country; ranking bars show source counts.")

    st.divider()
    st.subheader("A historical signal—not a climate attribution")
    fig=go.Figure(go.Scatter(
        x=annual.year,y=annual.vegetation_records,
        mode="lines+markers",line=dict(color=TEAL,width=3),
        marker=dict(size=6),
        hovertemplate="%{x}: %{y:,.0f} archived positives<extra></extra>"
    ))
    fig.add_annotation(x=int(years[0]),y=int(annual.vegetation_records.iloc[0]),
                       text="Partial first year",font=dict(color=AMBER,size=11),
                       showarrow=True,ax=55,ay=-40)
    if sensor=="MODIS" and 2002 in annual.year.values:
        aqua_year=annual.loc[annual.year.eq(2002)].iloc[0]
        fig.add_annotation(
            x=2002,y=int(aqua_year.vegetation_records),
            text="2001: Terra only; Aqua added in 2002",
            showarrow=True,ax=95,ay=-63,
            font=dict(color=BLUE,size=11)
        )
    last_year=int(annual.year.iloc[-1])
    last_months=annual.iloc[-1][MCOLS].to_numpy(dtype=float)
    available_months=int(np.count_nonzero(last_months>0))
    fig.add_annotation(
        x=last_year,y=int(annual.vegetation_records.iloc[-1]),
        text=f"{last_year}: {available_months} positive-record months; completeness unverified",
        showarrow=True,ax=-106,ay=-60,
        font=dict(color=AMBER,size=11)
    )
    style(fig,300,xaxis_title="Archive year",yaxis_title="Type-0 positive records")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Tracking one satellite product through time reveals shifts in its recorded activity; see Data provenance for measurement limits.")

    st.subheader(f"Global historical calendar · {years[0]}–{years[-1]}")
    vals=annual[MCOLS].to_numpy(dtype=float)
    percentages=100*vals/np.maximum(vals.sum(axis=1,keepdims=True),1)
    fig=go.Figure(go.Heatmap(
        x=MONTHS,y=annual.year.astype(str),z=percentages,customdata=vals,
        colorscale=SCALE,colorbar=dict(title="Yearly share (%)",thickness=12),
        hovertemplate="%{y} · %{x}<br>%{z:.1f}% of annual archived positives<br>"
                      "%{customdata:,.0f} records<extra></extra>"
    ))
    style(fig,480,yaxis=dict(autorange="reversed"),xaxis_title="UTC acquisition month")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Each year is normalized to 100% of its positive records, revealing seasonal timing. The first archive year is partial.")
    st.download_button("Export this year's country records",selected.to_csv(index=False),
                       f"MODVIIR_{sensor.lower()}_{year}_countries.csv","text/csv")
    with st.expander("Additional study · Amazon 2022–2025", expanded=False):
        st.write("Separate NASA MODIS + VIIRS S-NPP research from the Amazon study rectangle.")
        if st.checkbox("Open Amazon regional panel",value=False,key="v13_open_amazon"):
            from v10.amazon_section import render_amazon
            render_amazon()
    evidence_notes()


def render_country_compare():
    df, meta=load_world()
    headline(
        "02 / Compare Countries · Seasonal Fingerprints",
        "Find the rhythm behind regional hotspots",
        "Choose up to five countries, then compare trends and monthly **recorded "
        "activity fingerprints** using the same satellite product and time window."
    )
    sensor=selector("v11_compare_sensor")
    subset=df.loc[df.sensor.eq(sensor)].copy()
    years=sorted(subset.year.unique().tolist())
    start,end=st.slider(
        "Compare calendar years",min_value=years[0],max_value=years[-1],
        value=(2001,2025) if sensor=="MODIS" else (2019,2025),
        step=1,key="v11_compare_years"
    )
    subset=subset.loc[subset.year.between(start,end)]
    names=(subset.sort_values("year").drop_duplicates("iso3",keep="last")
           .set_index("iso3").country.to_dict())
    ids=sorted(names,key=lambda c:names[c].lower())
    defaults=[k for k in ("USA","BRA","AUS","COD","RUS") if k in ids]
    chosen=st.multiselect("Compare countries / territories (2–5)",
                           ids,default=defaults,
                           format_func=lambda v:names.get(v,v),
                           max_selections=5,key="v11_compare_countries")
    if len(chosen)<2:
        st.info("Select at least two countries to compare.")
        evidence_notes()
        return
    sub=subset.loc[subset.iso3.isin(chosen)].copy()
    sub["label"]=sub.iso3.map(names)
    x,y,z=st.columns(3)
    x.metric("Countries selected",str(len(chosen)))
    y.metric("Calendar years",str(end-start+1))
    z.metric("Type-0 records",help="Archived type-0 vegetation fire positive records",value=f"{int(sub.vegetation_records.sum()):,}")

    st.subheader("Country activity over time")
    show=st.radio("Chart mode",
                  ["Absolute archived records","Within-country indexed change"],
                  horizontal=True,key="v11_compare_chartmode")
    fig=go.Figure()
    for i,key in enumerate(chosen):
        rows=sub.loc[sub.iso3.eq(key)].sort_values("year")
        observed=rows.vegetation_records.to_numpy(dtype=float)
        if show.startswith("Within"):
            first=observed[np.flatnonzero(observed>0)[0]] if np.any(observed>0) else np.nan
            metric=100*observed/first if first>0 else np.full(len(observed),np.nan)
        else:
            metric=observed
        fig.add_trace(go.Scatter(
            x=rows.year,y=metric,mode="lines+markers",name=names[key],
            line=dict(width=2.6,color=COLOR[i%len(COLOR)]),
            customdata=observed,
            hovertemplate="%{x} · %{y:,.1f}<br>%{customdata:,.0f} archive records"
                          f"<extra>{names[key]}</extra>"
        ))
    style(fig,340,xaxis_title="Year",yaxis_title=(
        "Index: first nonzero year = 100" if show.startswith("Within") else
        "NASA presumed vegetation-fire records"
    ))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Switch between source counts and each country’s own indexed archive trajectory.")

    st.divider()
    st.subheader("Seasonal fingerprints · Relative monthly distribution")
    grouped=sub.groupby("iso3",as_index=True)[MCOLS].sum()
    profiles={}
    for key in chosen:
        values=grouped.loc[key,MCOLS].to_numpy(dtype=float)
        profiles[key]=values/values.sum() if values.sum()>0 else np.zeros(12)
    heat=np.vstack([profiles[v]*100 for v in chosen])
    fig=go.Figure(go.Heatmap(
        z=heat,y=[names[v] for v in chosen],x=MONTHS,
        colorscale=SCALE,colorbar=dict(title="Monthly share (%)",thickness=12),
        hovertemplate="%{y} · %{x}<br>%{z:.1f}% of country's type-0 records<extra></extra>"
    ))
    style(fig,245+len(chosen)*32,xaxis_title="UTC acquisition month",
          yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Each country sums to 100%; compare the timing of its monthly archived hotspot reports.")

    with st.expander("Calendar-month resemblance", expanded=False):
        pair_rows=[]
        for a,b in combinations(chosen,2):
            pa,pb=profiles[a],profiles[b]
            if pa.sum()==0 or pb.sum()==0:continue
            # Jensen–Shannon distance uses base-2; outputs range 0..1.
            similarity=max(0,1-float(jensenshannon(pa,pb,base=2)))
            pair_rows.append({
                "Country A":names[a],"Country B":names[b],
                "Calendar resemblance (0–1)":round(similarity,3),
                "Peak A":MONTHS[int(np.argmax(pa))],
                "Peak B":MONTHS[int(np.argmax(pb))],
                "Peak gap (months)":min(abs(int(np.argmax(pa))-int(np.argmax(pb))),12-abs(int(np.argmax(pa))-int(np.argmax(pb))))
            })
        if pair_rows:
            st.dataframe(pd.DataFrame(pair_rows).sort_values(
                "Calendar resemblance (0–1)",ascending=False),
                hide_index=True,use_container_width=True)
        st.caption(
            "Resemblance = 1 − Jensen–Shannon distance between 12 monthly "
            "record shares. This compares the **same calendar months**, "
            "not weather seasons or climate patterns. The peak-gap column "
            "makes timing differences visible."
        )
        if "USA" in profiles and "AUS" in profiles:
            us=profiles["USA"];au=profiles["AUS"]
            similarity_usa_aus=1-float(jensenshannon(us,au,base=2))
            st.info(
                f"**Why U.S. ↔ Australia ≈ {similarity_usa_aus:.3f}?** "
                f"National hotspot totals include many burning regimes, and "
                f"their monthly distributions overlap even across opposite hemispheres. "
                f"For these years, the U.S. peaks in "
                f"**{MONTHS[int(np.argmax(us))]}** and Australia in "
                f"**{MONTHS[int(np.argmax(au))]}**. "
                "This is shared monthly shape, not proof of equivalent fire seasons."
            )

    st.divider()
    st.subheader("One-country historical calendar")
    focus=st.selectbox("Inspect country",chosen,format_func=lambda c:names[c],
                       key="v11_country_focus")
    rows=sub.loc[sub.iso3.eq(focus)].sort_values("year")
    fig=go.Figure(go.Heatmap(
        z=rows[MCOLS].to_numpy(dtype=float),x=MONTHS,
        y=rows.year.astype(str),colorscale=SCALE,
        colorbar=dict(title="Records",thickness=12),
        hovertemplate="%{y} · %{x}<br>%{z:,.0f} archived positives<extra></extra>"
    ))
    style(fig,375,xaxis_title="Acquisition month",
          yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig,use_container_width=True)
    st.divider()
    st.subheader("Retrospective unusual-month screen")
    st.markdown(
        "Compare this country's monthly archive records against only **earlier "
        "years of the same calendar month**. This is exploratory evidence "
        "screening, not a statistically calibrated anomaly alert."
    )
    choice_year=st.selectbox(
        "Year to investigate",sorted(rows.year.unique().tolist(),reverse=True),
        index=0,key="v11_critical_year"
    )
    lookback=st.slider("Prior-year reference window",3,10,5,key="v11_critical_lookback")
    first_full=2001 if sensor=="MODIS" else 2019
    peers=rows.loc[rows.year.between(max(first_full,choice_year-lookback),
                                      choice_year-1)]
    target=rows.loc[rows.year.eq(choice_year)]
    if len(peers)<3 or target.empty:
        st.info("Fewer than three prior full-year references are available. "
                "Choose a later year or a longer reference window.")
    else:
        observed=target.iloc[0][MCOLS].to_numpy(dtype=float)
        older=peers[MCOLS].to_numpy(dtype=float)
        med=np.median(older,axis=0)
        upper=np.quantile(older,.90,axis=0)
        ratio=observed/np.maximum(med,1.)
        unusual=(observed>upper)&(ratio>=1.5)&(observed>=100)
        evidence=pd.DataFrame({
            "Month":MONTHS,"Observed archived positives":observed.astype(int),
            "Prior median":med.round(1),"Prior 90th percentile":upper.round(1),
            "Current / median":ratio.round(2),"Candidate":unusual
        })
        c1,c2=st.columns(2)
        c1.metric("High-activity months",help="Months flagged for high recorded positive activity",value=int(unusual.sum()))
        c2.metric("Reference years",help="Previous full calendar years used for comparison",value=len(peers))
        fig=go.Figure([
            go.Bar(x=MONTHS,y=observed,name=f"{choice_year} records",marker_color=AMBER),
            go.Scatter(x=MONTHS,y=med,name="Earlier-years median",mode="lines+markers",
                       line=dict(color=TEAL,width=2.5))
        ])
        style(fig,310,yaxis_title="NASA type-0 positive records")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(evidence,hide_index=True,use_container_width=True)
        st.caption(
            "Candidate rule: above historical same-month 90th percentile, ≥1.5× "
            "previous-years median, and ≥100 records; reference years: "+
            ", ".join(str(int(v)) for v in peers.year.tolist())+
            ". No satellite-coverage normalization or calibrated significance."
        )
        st.download_button("Export monthly screening audit",evidence.to_csv(index=False),
                           f"MODVIIR_{focus.lower()}_{choice_year}_screen.csv","text/csv")

    st.divider()
    st.subheader("Two-satellite geographic perspectives")
    st.write(
        "Compare each selected country's **share of the worldwide recorded "
        "positives of each separate sensor**. The shares do not represent "
        "calibrated fire rates or detection agreement."
    )
    overlap_year=st.select_slider("Common archive year",options=list(range(2019,2026)),
                                  value=2024,key="v11_overlap_year")
    both=df.loc[df.year.eq(overlap_year)&
                df.sensor.isin(["MODIS","VIIRS_NOAA20"])].copy()
    totals=both.groupby("sensor").vegetation_records.sum().to_dict()
    both=both.loc[both.iso3.isin(chosen)].copy()
    both["share_pct"]=both.apply(
        lambda r:100*r.vegetation_records/max(totals[r.sensor],1),axis=1
    )
    chart=go.Figure()
    for product,hue in [("MODIS",BLUE),("VIIRS_NOAA20",AMBER)]:
        subset=(both.loc[both.sensor.eq(product)].set_index("iso3")
                .reindex(chosen))
        chart.add_trace(go.Bar(
            x=[names[c] for c in chosen],y=subset.share_pct,
            marker_color=hue,name="MODIS" if product=="MODIS" else "VIIRS NOAA-20"
        ))
    style(chart,305,barmode="group",
          yaxis_title="Share of this sensor's worldwide positives (%)")
    st.plotly_chart(chart,use_container_width=True)
    st.caption("This visual normalizes the worldwide **composition within a "
               "sensor/year only**. Different sensor sampling and detection "
               "capabilities still apply. This is not a fire-severity or "
               "sensor-performance metric.")

    st.download_button("Export selected country's historical table",
                       rows.to_csv(index=False),
                       f"MODVIIR_{focus.lower()}_archive.csv","text/csv")
    evidence_notes()
