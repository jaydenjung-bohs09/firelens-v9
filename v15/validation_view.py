"""Judge-visible leave-one-year-out calibration evaluation; archived counts only."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from v15.calibration import country_month_rows
from v15.validation import (
    VALIDATION_CSV,validation_predictions,selected_method,summarize_predictions,
)

@st.cache_data(show_spinner="Reproducing held-out satellite record comparisons…")
def _out_of_sample_points():
    # One small, reproducible ~2-second analysis cached across app interactions.
    return validation_predictions(country_month_rows())

@st.cache_data(show_spinner=False)
def _summary():
    return pd.read_csv(VALIDATION_CSV)

def render_validation():
    st.subheader("Does calibration work? (held-out years)")
    st.caption(
        "NASA MODIS Terra/Aqua → VIIRS **NOAA-20** monthly country records; "
        "2019–2025 complete source-calendar overlap. This is an archival "
        "count mapping, not calibrated satellite observation coverage."
    )
    summary=_summary()
    global_results=summary.loc[summary.level.eq("global")].copy()
    selected=selected_method(summary)
    human={"none":"Unadjusted MODIS","ratio":"Monthly country ratio",
           "log":"Region-shrunk log regression","global_ratio":"Single global ratio"}
    label=human[selected]
    st.write(
        f"**Lowest held-out median absolute percentage error: {label}.** "
        "For each validation year, all other overlap years train the model; "
        "the withheld year alone supplies its evaluation records."
    )
    continent=st.selectbox(
        "Results for",["GLOBAL"]+sorted(c for c in summary.continent.unique() if c!="GLOBAL"),
        key="v15_validation_region"
    )
    subset=summary.loc[summary.continent.eq(continent)].copy()
    table=subset[["method","n_predictions","n_modis_zero","median_abs_pct_error",
                  "median_bias_pct","spearman"]].rename(columns={
        "method":"Candidate method","n_predictions":"Evaluated country-months",
        "n_modis_zero":"MODIS-zero (undefined estimates)",
        "median_abs_pct_error":"Median absolute % error",
        "median_bias_pct":"Median bias (%)","spearman":"Spearman ρ"
    })
    table["Candidate method"]=table["Candidate method"].map(human)
    for col in ("Median absolute % error","Median bias (%)","Spearman ρ"):
        table[col]=table[col].round(3)
    st.dataframe(table,hide_index=True,use_container_width=True)
    if True:  # Scatter is contained in the Global Atlas proof expander.
        method=st.radio("Conversion",list(human),
                        format_func=lambda m:human[m],horizontal=True,
                        index=list(human).index(selected),key="v15_scatter_method")
        all_points=_out_of_sample_points()
        if continent!="GLOBAL":
            all_points=all_points.loc[all_points.continent.eq(continent)]
        years=sorted(all_points.year.unique().tolist())
        year=st.selectbox("Withheld year",["All"]+years,
                          key="v15_scatter_year")
        if year!="All":
            all_points=all_points.loc[all_points.year.eq(year)]
        values=all_points.loc[
            all_points.method.eq(method)&all_points.predicted.gt(0)&
            all_points.viirs.gt(0)
        ]
        # Deterministic sampling for responsiveness; all metrics above use
        # every eligible held-out month, not the scatter sample.
        if len(values)>2000:
            values=values.sample(n=2000,random_state=20261115).sort_values(
                ["year","iso3","month"])
        custom=np.stack([values.iso3.astype(str),
                         values.year.astype(str),
                         values.month.astype(str)],axis=-1)
        fig=go.Figure(go.Scattergl(
            x=values.viirs,y=values.predicted,mode="markers",
            marker=dict(size=5,color="#39D8C2",opacity=.5),
            customdata=custom,
            hovertemplate="Country: %{customdata[0]} · %{customdata[1]}-%{customdata[2]}"
                         "<br>Held-out NOAA-20: %{x:,.0f}"
                         "<br>MODIS-based estimate: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ))
        if not values.empty:
            lower=max(.5,min(values.viirs.min(),values.predicted.min()))
            upper=max(values.viirs.max(),values.predicted.max())*1.2
            fig.add_trace(go.Scatter(
                x=[lower,upper],y=[lower,upper],
                mode="lines",line=dict(color="#FFAE65",dash="dash",width=2),
                name="Perfect agreement"
            ))
        fig.update_layout(
            template="plotly_dark",height=410,
            title="Withheld year: estimated vs observed monthly records",
            xaxis=dict(type="log",title="Actual VIIRS NOAA-20 positive records"),
            yaxis=dict(type="log",title="MODIS-based estimated NOAA-20 records"),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,33,51,.8)",
            margin=dict(l=40,r=20,t=60,b=45),
        )
        st.plotly_chart(fig,use_container_width=True)
        st.caption(
            "Scatter shows up to 2,000 seeded positive/positive country-months; "
            "the table uses all eligible pairs. When MODIS has zero reports, "
            "its estimate is undefined and is excluded—not set to zero. "
            "Percentage errors also exclude actual NOAA-20 zeros."
        )
    st.download_button(
        "Download full held-out validation table",
        summary.to_csv(index=False),
        file_name="MODVIIR_Held_Out_Calibration_Validation.csv",
        mime="text/csv",
        key="v15_validation_download",
    )
