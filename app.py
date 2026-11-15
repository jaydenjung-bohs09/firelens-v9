"""FireLens V11: Global Atlas, Country Comparison and California Fire Laboratory.

Run: streamlit run app.py
Data: NASA FIRMS archives; all results are detection proxies, NOT fire truth.
This new entry point does not modify the frozen legacy_v8 application.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pyproj import Transformer
from critical_periods import seasonal_screen

ROOT = Path(__file__).resolve().parent
HIST = ROOT / "data" / "historic"
PRED = ROOT / "data" / "predict"
BG = "#081322"
PANEL = "#0F2133"
TXT = "#E8F0F6"
MUTED = "#9CAFC0"
ORANGE = "#FF9E54"
TEAL = "#39D8C2"
BLUE = "#77ACF8"
RED = "#FA6C75"
PALETTE = ["#101F2B", "#15394B", "#236576", "#C98A43", "#F2B36E", "#FA7355"]

st.set_page_config(page_title="FireLens | NASA Hotspot Evidence", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(155deg,#071422 0%,#081B28 55%,#0A1422 100%);}
[data-testid="stSidebar"]{background:#091927;min-width:270px;}
.block-container{padding-top:3.75rem;padding-bottom:2.3rem;max-width:1450px;}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{font-size:1.08rem;line-height:1.58;}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p{font-size:.95rem!important;color:#C5D4E1!important;line-height:1.5!important;}
[data-testid="stWidgetLabel"] p{font-size:1.03rem!important;font-weight:650!important;color:#E8F0F6!important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2{font-size:1.65rem!important;color:#F7FCFF!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-size:1.07rem!important;font-weight:700!important;color:#E4F3F8!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{gap:.62rem!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.33rem 0!important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.70rem!important;}
[data-testid="stSidebar"] hr{margin:.55rem 0!important;}
button[kind="primary"],button[kind="secondary"]{font-size:1rem!important;font-weight:680!important;}
[data-testid="stMetric"]{background:#0F2332;border:1px solid #345469;border-radius:12px;padding:.9rem;}
[data-testid="stMetricLabel"]{color:#CEE1EF;font-size:.98rem!important}
[data-testid="stMetricValue"]{color:#FFFFFF;font-weight:770;font-size:1.68rem!important}
[data-testid="stMetricDelta"]{color:#71E6CA}
button[data-baseweb="tab"]{font-weight:680;font-size:1.06rem}
@media (max-width:750px){
 .block-container{padding-left:1rem;padding-right:1rem;}
 [data-testid="stMetricValue"]{font-size:1.35rem!important;}
}
div.stAlert{border-radius:12px;}
a{color:#50DCC2!important}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_manifest():
    m=pd.read_csv(PRED/'sequence_manifest.csv')
    m['target_date_utc']=pd.to_datetime(m['target_date_utc'],errors='raise',utc=True).dt.tz_convert(None)
    m['year']=m.target_date_utc.dt.year
    m['month']=m.target_date_utc.dt.month
    m['day_name']=m.target_date_utc.dt.day_name()
    m['both_recorded']=m['target_both_sensors_have_any_hotspot_record'].astype(bool)
    m['none_recorded']=~(m['target_viirs_has_any_hotspot_record'].astype(bool)|m['target_modis_has_any_hotspot_record'].astype(bool))
    m['one_sided_record_day']=~m['both_recorded'] & ~m['none_recorded']
    return m

@st.cache_data(show_spinner=False)
def load_model_data():
    with np.load(PRED/'test_predictions.npz',allow_pickle=False) as z:
        scores=np.array(z['predicted_detection_proxy_scores'])  # preserve float16 threshold semantics used in source evaluation
        actual=np.array(z['actual_recorded_hotspot_labels'],dtype=bool)
        dates=[str(v) for v in z['target_dates']]
    metrics=pd.read_csv(PRED/'model_metrics.csv')
    baseline=pd.read_csv(PRED/'baseline_metrics.csv')
    model_card=json.loads((PRED/'model_card.json').read_text())
    ds_card=json.loads((PRED/'dataset_card.json').read_text())
    return scores, actual, dates, metrics, baseline, model_card, ds_card

@st.cache_data(show_spinner=False)
def load_park():
    groups=pd.read_csv(HIST/'park_evidence_groups.csv')
    rec=pd.read_csv(HIST/'park_hotspot_records.csv')
    daily=pd.read_csv(HIST/'july_2024_sensor_status.csv')
    geom=json.loads((HIST/'park_final_historical_USER_EXPORT.geojson').read_text())['features'][0]['geometry']
    return groups,rec,daily,geom

@st.cache_data(show_spinner=False)
def load_july():
    df=pd.read_csv(HIST/'california_july_2024_hotspots.csv')
    df['date']=pd.to_datetime(df['date'])
    return df

def theme(fig, height=350, **kwargs):
    fig.update_layout(template='plotly_dark',height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(10,25,38,.65)',font_color=TXT,
                      margin=dict(l=35,r=20,t=48,b=45),hoverlabel=dict(bgcolor=PANEL,font_size=12),**kwargs)
    return fig

def paragraph(text):
    st.markdown(f"<p style='color:#C7D7E4;font-size:1.13rem;line-height:1.6;max-width:1000px'>{text}</p>",unsafe_allow_html=True)

def section_eyebrow(text):
    st.markdown(f"<div style='font-size:.87rem;font-weight:830;letter-spacing:.12em;color:{TEAL};text-transform:uppercase;margin-bottom:.3rem'>{text}</div>",unsafe_allow_html=True)

def counts(pred, truth):
    tp=int(np.logical_and(pred,truth).sum());fp=int(np.logical_and(pred,~truth).sum());fn=int(np.logical_and(~pred,truth).sum())
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1)

def source_footer():
    st.caption("NASA FIRMS historical records · Detailed methods and interpretation limits are in Methods & Sources.")


def _jump_to_california(section):
    """Navigate before Streamlit instantiates sidebar widgets on rerun."""
    st.session_state["v11_workspace"]="03 · CALIFORNIA FIRE LABORATORY"
    st.session_state["v11_california_tool"]=section


# Three primary workspaces are intentionally routed before loading California's
# frozen model arrays. Global browsing stays fast and memory-conscious.
with st.sidebar:
    st.markdown("## FIRELENS")
    st.caption("NASA SATELLITE EVIDENCE")
    workspace = st.radio(
        "MAIN SECTIONS",
        ["01 · GLOBAL ATLAS", "02 · COUNTRY COMPARISON", "03 · CALIFORNIA FIRE LABORATORY"],
        key="v11_workspace"
    )
    st.divider()
    st.markdown("**Follow the evidence**")
    st.caption("Global → Country → Park Fire → Forecast")
    st.markdown(
        "[NASA data](https://firms.modaps.eosdis.nasa.gov/download/) · "
        "[Challenge](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)"
    )

if workspace.startswith("01"):
    st.info(
        "**Explore the whole planet—or follow the Park Fire case study.** "
        "FireLens moves from recorded hotspots to a documented California "
        "satellite-data gap and an experimental model comparison."
    )
    st.button("Follow the flagship story: Park Fire → California",
              key="v13_global_story",type="primary",
              on_click=_jump_to_california,args=("02  INVESTIGATE EVIDENCE",))
    from v11.world import render_global_atlas
    render_global_atlas()
    st.stop()
if workspace.startswith("02"):
    st.button("Continue to Park Fire evidence",key="v13_country_story",
              on_click=_jump_to_california,args=("02  INVESTIGATE EVIDENCE",))
    from v11.world import render_country_compare
    render_country_compare()
    st.stop()

# Verify frozen model evidence at startup. Never silently display incompatible arrays.
manifest=load_manifest()
scores, actual, dates, metrics, baselines, mcard, dcard=load_model_data()
threshold=float(mcard['selected_threshold_on_validation_only'])
pooled=counts(scores>=threshold,actual)
if (pooled['tp'],pooled['fp'],pooled['fn']) != (3581,4363,5480):
    st.error('The frozen prediction archive failed integrity checks. Do not cite model results.')
    st.stop()
if len(manifest)!=1454 or manifest.split.value_counts().to_dict()!={'train':870,'validation':292,'test':292}:
    st.error('Sequence manifest failed 60/20/20 date split checks.')
    st.stop()

section_eyebrow('NASA SPACE APPS CHALLENGE 2026 / HARMONIZING MODIS & VIIRS HOTSPOTS')
st.markdown("<h1 style='font-size:3.2rem;letter-spacing:-.04em;margin:.12rem 0 .32rem'>FireLens <span style='color:#FFAE65'>/ California</span></h1>",unsafe_allow_html=True)
st.markdown("<div style='font-size:1.38rem;color:#E4F1F8;font-weight:620;margin-bottom:.7rem'>California Fire Laboratory · From orbit to evidence</div>",unsafe_allow_html=True)
paragraph('One fire, two satellite records, one critical question: did a missing detection mean less burning or missing evidence? Investigate the July 2024 Park Fire and test whether historical hotspot patterns improve next-day recorded-cell forecasts.')
st.markdown("### Choose a California investigation")
section=st.radio(
    'California research tools',
    ['02  INVESTIGATE EVIDENCE','03  REPLAY THE MODEL',
     '01  EXPLORE HISTORY','METHODS & SOURCES'],
    format_func=lambda v:{
        '02  INVESTIGATE EVIDENCE':'Park Fire Evidence',
        '03  REPLAY THE MODEL':'AI Forecast Comparison',
        '01  EXPLORE HISTORY':'Historical Calendar',
        'METHODS & SOURCES':'Methods & Sources',
    }[v],
    horizontal=True,
    label_visibility='collapsed',
    key='v11_california_tool',
)
st.divider()

k1,k2,k3,k4=st.columns(4)
k1.metric('CA archive records','173,471',help='California training-archive records')
k2.metric('Park Fire positive groups','316')
k3.metric('Time-aligned groups','45 / 316',help='Time-compatible positive groups in the Park Fire reference audit')
k4.metric('California ConvLSTM F1','0.4212')
st.caption('Comparison reference: persistence F1 0.3818.')
st.caption('Intended user: a wildfire intelligence analyst who needs to separate satellite reporting gaps from real changes in recorded activity before interpreting historical patterns. Forecast scores come from a separate 2025 California archive experiment.')


if section == '01  EXPLORE HISTORY':
    section_eyebrow('Observe / Four-year calendar')
    st.header('Four years of recorded satellite activity')
    paragraph('This historical overview counts 10 km cells with at least one positive MODIS or VIIRS hotspot record per UTC day. It uses the frozen training dataset manifest—not inferred burned area or a validated sensor-normalized fire rate.')
    m=manifest.copy()
    month_grid=m.groupby(['year','month'])['target_positive_cells'].sum().unstack('month').reindex(index=[2022,2023,2024,2025],columns=list(range(1,13)))
    z=month_grid.to_numpy(dtype=float)
    fig=go.Figure(go.Heatmap(z=z,x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],y=['2022','2023','2024','2025'],
                             colorscale=[[0,'#102A37'],[.2,'#1D6A75'],[.52,'#F6A653'],[1,'#F05A4D']],
                             colorbar=dict(title='Cell-days'),hovertemplate='%{y} %{x}<br>%{z:,.0f} occupied grid cell-days<extra></extra>'))
    fig.update_yaxes(autorange='reversed')
    theme(fig,270,title='Monthly sum of occupied 10 km cell-days (detection proxy)')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('The calendar begins January 8, 2022 because each next-day target needs seven prior days.')
    a,b=st.columns([1,3])
    with a:
        yr=st.selectbox('Explore year',[2022,2023,2024,2025],index=2)
        only_both=st.checkbox('Show days with hotspot records from both sensors only',value=False)
        st.caption('Zero here means no archived positive records on that date.')
    with b:
        local=m[m.year==yr].copy()
        if only_both:
            local.loc[~local.both_recorded,'target_positive_cells']=np.nan
        daily_fig=go.Figure()
        daily_fig.add_trace(go.Scatter(x=local.target_date_utc,y=local.target_positive_cells,mode='lines',name='Occupied cells',line=dict(color=TEAL,width=2),fill='tozeroy',fillcolor='rgba(57,216,194,.10)'))
        bad=local[(local.none_recorded)|(local.one_sided_record_day)|(local.target_snpp_known_anomaly_window)]
        if len(bad):
            daily_fig.add_trace(go.Scatter(x=bad.target_date_utc,y=bad.target_positive_cells.fillna(0),mode='markers',name='Record-availability caveat',marker=dict(color=ORANGE,size=6,symbol='x')))
        theme(daily_fig,290,title=f'{yr} daily recorded activity and missing-evidence warnings',xaxis_title='UTC date',yaxis_title='Positive 10 km cells')
        st.plotly_chart(daily_fig,use_container_width=True)
    mask=m[m.year==yr]
    q1,q2,q3=st.columns(3)
    q1.metric('Recorded occupied cell-days',f'{int(mask.target_positive_cells.sum()):,}')
    q2.metric('Days with no VIIRS records',str(int((~mask.target_viirs_has_any_hotspot_record).sum())))
    q3.metric('Days with no MODIS records',str(int((~mask.target_modis_has_any_hotspot_record).sum())))
    st.caption('Cross-sensor date flags summarize archived positive-report availability; consult Methods & Sources for observation-coverage limitations.')
    st.divider()
    # Descriptive retrospective peer-year screening; separate from AI model.
    st.divider()
    section_eyebrow('Discover / Retrospective seasonal comparisons')
    st.subheader('Compare archived activity with historical seasonal patterns')
    paragraph('Compare each day with the same season in the other three years (±14 calendar days). The screen uses only dates with a positive MODIS AND VIIRS record and without a documented S-NPP anomaly-window flag. This is a descriptive historical screening tool, not wildfire identification, statistical significance, or a live alert.')
    x1,x2,x3=st.columns(3)
    with x1:
        min_run=st.slider('Minimum consecutive candidate days',1,7,value=2,step=1,key='critical_run')
    with x2:
        ratio=st.slider('Minimum current / peer median',1.0,3.0,value=1.25,step=.05,key='critical_ratio')
    with x3:
        peer_q=st.select_slider('Historical reference percentile',options=[0.75,0.80,0.85,0.90,0.95],value=0.90,key='critical_quantile')
    critical_daily,critical_periods=seasonal_screen(
        m,window_days=14,peer_quantile=peer_q,
        ratio_threshold=ratio,min_occupied_cells=20,min_peers=30,min_streak=min_run)
    dyear=critical_daily[critical_daily.year==yr].copy()
    # Empty screen results must not invoke .dt on object-typed columns.
    if critical_periods.empty:
        periods_year=critical_periods.copy()
    else:
        starts=pd.to_datetime(critical_periods['start_date'],errors='coerce',utc=True)
        periods_year=critical_periods.loc[starts.dt.year.eq(yr)].copy()
    # Evidence-review candidates are deliberately separate from the
    # comparable activity-anomaly screen: they reveal missing observation
    # context, NOT anomalous fire intensity or a wildfire prediction.
    gap_review=dyear.loc[
        dyear['target_snpp_known_anomaly_window'].astype(bool)
        & dyear['target_modis_has_any_hotspot_record'].astype(bool)
        & ~dyear['target_viirs_has_any_hotspot_record'].astype(bool)
    ].copy()
    r1,r2,r3,r4=st.columns(4)
    r1.metric('Candidate activity periods',len(periods_year))
    r2.metric('Days not comparable',int((~dyear.comparison_eligible).sum()))
    r3.metric('Candidate activity days',int(dyear.candidate_day.sum()))
    r4.metric('S-NPP data-gap review days',len(gap_review))
    trace=go.Figure()
    trace.add_trace(go.Scatter(x=dyear.date,y=dyear.occupied_cells,name='Recorded occupied cells',mode='lines',line=dict(color=TEAL,width=2.4)))
    trace.add_trace(go.Scatter(x=dyear.date,y=dyear.peer_median,name='Other-years seasonal median',mode='lines',line=dict(color=BLUE,width=1.8,dash='dash')))
    trace.add_trace(go.Scatter(x=dyear.date,y=dyear.peer_upper_quantile,name=f'Other-years {int(peer_q*100)}th percentile',mode='lines',line=dict(color=ORANGE,width=1.5,dash='dot')))
    limited=dyear[~dyear.comparison_eligible]
    if len(limited):
        trace.add_trace(go.Scatter(x=limited.date,y=limited.occupied_cells,mode='markers',name='Comparison limited',text=limited.record_context,hovertemplate='%{x|%Y-%m-%d}<br>%{text}<br>Positive cells %{y}<extra></extra>',marker=dict(color=RED,size=6,symbol='x')))
    if not gap_review.empty:
        trace.add_trace(go.Scatter(
            x=gap_review.date,y=gap_review.occupied_cells,
            mode='markers',name='S-NPP data-gap review (separate from anomaly)',
            marker=dict(color=RED,size=12,symbol='triangle-up',
                        line=dict(color='#EAF4F8',width=.7)),
            hovertemplate='%{x|%Y-%m-%d}<br>Known S-NPP issue'
                          '<br>MODIS source records present'
                          '<br>Recorded union cells %{y:,}<extra></extra>'
        ))
    for _,p0 in periods_year.iterrows():
        trace.add_vrect(x0=p0.start_date,x1=p0.end_date+pd.Timedelta(days=1),fillcolor=TEAL,opacity=.12,line_width=0)
    theme(trace,345,title=f'{yr} recorded activity vs other-year seasonal reference',xaxis_title='UTC target date',yaxis_title='Occupied 10 km grid cells')
    st.plotly_chart(trace,use_container_width=True)
    if yr == 2024:
        july_window=dyear.loc[dyear.date.between("2024-07-24","2024-07-28")]
        flagged=int(july_window["target_snpp_known_anomaly_window"].sum())
        eligible=int(july_window["comparison_eligible"].sum())
        st.info(
            "**Why doesn't the Park Fire appear as a candidate?** The screen "
            "compares *statewide* positive 10-km cell counts, not individual "
            "wildfire onset. The July 24–28, 2024 S-NPP disruption overlaps "
            f"the Park Fire's growth; {flagged} of those 5 days are flagged "
            f"and only {eligible} pass the two-sensor comparison filter. "
            "They are deliberately excluded from the activity-anomaly "
            "ranking but **shown separately as data-gap review markers**. "
            "This is an evidence-quality safeguard, not evidence that the "
            "fire was minor. Inspect it under **Park Fire Evidence**."
        )
    if len(periods_year):
        labels=[f"{row.start_date:%b %d}–{row.end_date:%b %d, %Y} · {int(row.days)} day(s)" for row in periods_year.itertuples()]
        selected_index=st.selectbox('Inspect candidate period',list(range(len(periods_year))),format_func=lambda i: labels[i],key='selected_critical')
        chosen=periods_year.iloc[selected_index]
        st.info(f"Selected period: **{chosen.start_date:%Y-%m-%d} to {chosen.end_date:%Y-%m-%d}**. Peak **{chosen.peak_occupied_cells:.0f}** recorded occupied 10 km cells on {chosen.peak_date:%Y-%m-%d}; **{chosen.peak_ratio:.2f}×** the comparable peer-year seasonal median at peak. This does not identify a particular wildfire or establish a sensor-normalized increase.")
        show=periods_year[['start_date','end_date','days','peak_occupied_cells','sum_excess_cell_days','peak_ratio']].copy()
        for date_col in ('start_date','end_date'):
            show[date_col]=pd.to_datetime(show[date_col]).dt.strftime('%b %d, %Y')
        st.dataframe(show.rename(columns={
            "start_date":"Start (UTC)",
            "end_date":"End (UTC)",
            "days":"Consecutive days",
            "peak_occupied_cells":"Peak positive cells",
            "sum_excess_cell_days":"Excess cell-days vs seasonal median",
            "peak_ratio":"Peak relative to seasonal median",
        }),hide_index=True,use_container_width=True,height=225)
    else:
        st.info('No consecutive periods meet these descriptive settings in the selected year. You can change the screening controls.')
    cexp1,cexp2=st.columns(2)
    with cexp1:
        st.download_button('Export candidate periods CSV',critical_periods.to_csv(index=False),file_name='firelens_v91_candidate_periods.csv',mime='text/csv')
    with cexp2:
        st.download_button('Export evidence-aware daily screen',critical_daily.to_csv(index=False),file_name='firelens_v91_daily_screen.csv',mime='text/csv')
    st.caption('Retrospective screening: ±14 calendar days in other archive years, ≥30 comparable reference days, ≥20 positive cells, configurable percentile, median ratio and duration. See Methods & Sources for inference limits.')
    st.divider()
    section_eyebrow('Study area / California observations')
    st.subheader('Explore a selected area: July 2024')
    paragraph('For an arbitrary geographic rectangle, this view uses the actual July 2024 FIRMS geolocations rather than the 4-year aggregated sequence manifest. Change the study area to see the observed daily cell counts.')
    july=load_july()
    preset=st.radio('Area', ['California archive extent','Park Fire vicinity (~60 km screening rectangle)','Brea / North Orange County vicinity','Custom rectangle'],horizontal=True,index=1)
    if preset=='California archive extent':
        lat_lo,lat_hi,lon_lo,lon_hi=32.5,42.1,-124.5,-114.0
    elif preset.startswith('Park Fire'):
        lat_lo,lat_hi,lon_lo,lon_hi=39.2,40.3,-122.5,-121.0
    elif preset.startswith('Brea'):
        lat_lo,lat_hi,lon_lo,lon_hi=33.65,34.10,-118.25,-117.40
    else:
        u,v=st.columns(2)
        with u:
            lat_lo,lat_hi=st.slider('Latitude (degrees N)',32.5,42.1,(39.2,40.3),step=.1)
        with v:
            lon_lo,lon_hi=st.slider('Longitude (degrees E, negative)',-124.5,-114.0,(-122.5,-121.0),step=.1)
    region=july[july.latitude.between(lat_lo,lat_hi)&july.longitude.between(lon_lo,lon_hi)].copy()
    trans=Transformer.from_crs('EPSG:4326','EPSG:6933',always_xy=True)
    if len(region):
        x,y=trans.transform(region.longitude.to_numpy(),region.latitude.to_numpy())
        region['cx']=np.floor(np.array(x)/5000).astype(int)
        region['cy']=np.floor(np.array(y)/5000).astype(int)
        region['cell']=region['cx'].astype(str)+'_'+region['cy'].astype(str)
        active=region.groupby('date')['cell'].nunique().reindex(pd.date_range('2024-07-01','2024-07-31'),fill_value=0)
        cnt=region.groupby(['date','sensor_group'])['cell'].nunique().unstack(fill_value=0).reindex(pd.date_range('2024-07-01','2024-07-31'),fill_value=0)
        for s in ('MODIS','VIIRS'):
            if s not in cnt.columns:cnt[s]=0
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=active.index,y=active.values,name='Union occupied cells',line=dict(color=TEAL,width=3)))
        fig.add_trace(go.Scatter(x=cnt.index,y=cnt.MODIS,name='MODIS positive cells',line=dict(color=BLUE,dash='dot')))
        fig.add_trace(go.Scatter(x=cnt.index,y=cnt.VIIRS,name='VIIRS positive cells',line=dict(color=ORANGE,dash='dot')))
        if preset.startswith('Park Fire'):
            # NASA's reported S-NPP CDP disruption began July 24, 05:28 UTC.
            # July 25–28 are listed as missing for VIIRS S-NPP in FIRMS.
            fig.add_vrect(x0='2024-07-24',x1='2024-07-29',
                          fillcolor='rgba(250,108,117,.20)',
                          opacity=.5,line_width=0,
                          annotation_text='S-NPP data disruption',
                          annotation_position='top left')
        theme(fig,305,title='Selected area · July 2024 · 5 km grid',xaxis_title='UTC date',yaxis_title='Positive occupied cells')
        st.plotly_chart(fig,use_container_width=True)
        if preset.startswith('Park Fire'):
            st.warning(
                "**NASA-documented satellite data gap:** VIIRS S-NPP "
                "has no positive archive records in this area on July 24–28, "
                "while MODIS recorded extensive activity during July 25–28. "
                "NASA reports an S-NPP command-processor disruption beginning "
                "**July 24 at 05:28 UTC**; FIRMS lists **July 25–28** as "
                "missing VIIRS S-NPP archive dates. Positive records resume "
                "in the selected area on **July 29**. This is not evidence "
                "that the fire disappeared. "
                "[NASA outage record](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html) · "
                "[NASA FIRMS missing dates](https://firms.modaps.eosdis.nasa.gov/api/missing_data/)"
            )
            status_park=pd.read_csv(HIST/'july_2024_sensor_status.csv')
            status_park=status_park.loc[status_park['UTC date'].between('2024-07-24','2024-07-30')]
            with st.expander('Examine the affected dates and statewide archive counts'):
                st.dataframe(status_park[[
                    'UTC date','MODIS hotspot records','VIIRS hotspot records',
                    'S-NPP anomaly window','FIRMS listed VIIRS missing date'
                ]].rename(columns={
                    'UTC date':'UTC date',
                    'MODIS hotspot records':'Statewide MODIS records',
                    'VIIRS hotspot records':'Statewide VIIRS S-NPP records',
                    'S-NPP anomaly window':'NASA platform anomaly flagged',
                    'FIRMS listed VIIRS missing date':'FIRMS missing date flagged'
                }),hide_index=True,use_container_width=True)
    else:
        st.warning('No positive hotspot records in the selected area and month. This is not proof there were no fires.')
    st.caption(f'{len(region):,} NASA FIRMS hotspot records in this rectangular study area; 5-km grid overlay.')
    source_footer()

if section == '02  INVESTIGATE EVIDENCE':
    section_eyebrow('Interpret / Park Fire evidence investigation')
    st.header('The missing data that changes the story')
    paragraph('July 2024 provides a case study in why raw sensor disagreements can mislead. FireLens compares positive-only cell/day/day-night groups within the final historical Park Fire footprint, then checks temporal compatibility and the documented S-NPP disruption window.')
    groups,parkrec,status,geometry=load_park()
    st.markdown('### What NASA recorded during the Park Fire')
    st.write(
        "The Park Fire began on July 24, 2024. In our NASA archive, "
        "MODIS reports many positives while VIIRS S-NPP has a documented "
        "data interruption. The chart below uses the **Park Fire vicinity "
        "screening rectangle**, not the eventual fire footprint."
    )
    july_park=load_july()
    vicinity=july_park.loc[
        july_park.latitude.between(39.2,40.3)
        & july_park.longitude.between(-122.5,-121.0)
        & july_park.date.between('2024-07-24','2024-07-31')
    ].copy()
    by_day=(vicinity.groupby(['date','sensor_group']).size()
            .unstack(fill_value=0)
            .reindex(pd.date_range('2024-07-24','2024-07-31'),fill_value=0))
    for sensor_name in ('MODIS','VIIRS'):
        if sensor_name not in by_day:
            by_day[sensor_name]=0
    timeline=go.Figure()
    timeline.add_trace(go.Bar(
        x=by_day.index,y=by_day.MODIS,name='MODIS (Terra/Aqua)',
        marker_color=BLUE
    ))
    timeline.add_trace(go.Bar(
        x=by_day.index,y=by_day.VIIRS,name='VIIRS S-NPP',
        marker_color=ORANGE
    ))
    timeline.add_vrect(
        x0='2024-07-24',x1='2024-07-29',
        fillcolor='rgba(250,108,117,.18)',opacity=.5,line_width=0,
        annotation_text='S-NPP data gap',annotation_position='top left'
    )
    theme(timeline,320,title='Park Fire vicinity · archived positive records by UTC date',
          barmode='group',xaxis_title='Date (UTC)',yaxis_title='Raw satellite hotspot records')
    st.plotly_chart(timeline,use_container_width=True)
    st.warning(
        "**Evidence gap, not a disappearance of fire:** NASA documents "
        "an S-NPP processing interruption beginning July 24 at 05:28 UTC, "
        "and its FIRMS missing-data list explicitly includes July 25–28. "
        "VIIRS S-NPP archive positives return here on July 29. "
        "[NASA outage notice](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html) · "
        "[FIRMS missing dates](https://firms.modaps.eosdis.nasa.gov/api/missing_data/)"
    )
    with st.expander('Show exact Park Fire vicinity source counts'):
        st.dataframe(by_day.reset_index().rename(columns={
            'index':'UTC date','MODIS':'MODIS records','VIIRS':'VIIRS S-NPP records'
        }).assign(**{'UTC date':lambda df:df['UTC date'].dt.strftime('%b %d, %Y')}),
        hide_index=True,use_container_width=True)
        st.caption('Counts are archived positives within 39.2–40.3°N, 122.5–121°W; they are not burned area or a daily mapped perimeter.')
    st.divider()
    from v12.evidence_inspector import render_inspector
    render_inspector(parkrec, status, groups, geometry)
    st.divider()
    st.markdown("### Reference audit · fixed three-hour rule")
    st.caption("The reference results below use the unchanged July 2024 Park Fire archive and the independently preserved grouping method. They are independent of the inspector sliders above.")
    p1,p2,p3=st.columns(3)
    p1.metric('Both positive · of 316 groups','54')
    p2.metric('Within 3h · of 54 both-positive','45')
    p3.metric('Within 3h · of all 316','45')
    st.caption('54/316 = 17.09% of positive groups; 45/54 = 83.33% conditional; 45/316 = 14.24% of positive groups.')
    st.warning('None of these percentages measure satellite sensitivity, operational coverage, accuracy, or observed no-fire locations. The final Park Fire polygon is retrospective, not the daily fire boundary.')
    categories=groups['Evidence classification'].value_counts().reset_index()
    categories.columns=['Evidence class','Groups']
    order=['Both positive; time-compatible','Both positive; no pair within tolerance','MODIS positive; VIIRS missing-data flag','MODIS positive only; VIIRS opportunity unknown','VIIRS positive only; MODIS opportunity unknown']
    colors={order[0]:TEAL,order[1]:ORANGE,order[2]:RED,order[3]:BLUE,order[4]:'#C093EF'}
    fig=go.Figure(go.Bar(y=[x[:46] for x in order][::-1],x=[int(categories.set_index('Evidence class').Groups.get(x,0)) for x in order][::-1],orientation='h',marker_color=[colors[x] for x in order][::-1],text=[int(categories.set_index('Evidence class').Groups.get(x,0)) for x in order][::-1],textposition='outside'))
    theme(fig,350,title='Every group is positive in at least one sensor',xaxis_title='Positive-record cell/day/day-night groups',showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
    dd=st.select_slider('Inspect UTC date',[str(x) for x in pd.date_range('2024-07-15','2024-07-31').strftime('%Y-%m-%d')],value='2024-07-24')
    selection=groups[groups['UTC date']==dd]
    mapcol,detailcol=st.columns([1.55,1])
    with mapcol:
        fig=go.Figure()
        # Original polygon geometry, rendered as coastline-like outline in lon/lat; no approximate replacement.
        polycoords=geometry['coordinates']
        for poly_i, polygon in enumerate(polycoords if geometry['type']=='MultiPolygon' else [polycoords]):
            ring=polygon[0]
            # Use a light geometry simplification only for plotting; the actual spatial filter used full resolution.
            step=max(1,len(ring)//450)
            pts=ring[::step]
            if pts[-1]!=ring[-1]:pts.append(ring[-1])
            fig.add_trace(go.Scattergl(x=[p[0] for p in pts],y=[p[1] for p in pts],mode='lines',line=dict(color='#CFB78A',width=1.2),name='Retrospective final footprint',hoverinfo='skip',showlegend=(poly_i==0)))
        if len(selection):
            for name in order:
                d=selection[selection['Evidence classification']==name]
                if len(d):
                    fig.add_trace(go.Scattergl(x=d.cell_center_lon,y=d.cell_center_lat,mode='markers',name=name[:35],marker=dict(size=9,color=colors[name],opacity=.92,line=dict(color='#0D1F2A',width=.5)),customdata=np.stack([d['Cell ID'].astype(str),d['Day/night'].astype(str)],axis=-1),hovertemplate='Grid: %{customdata[0]}<br>UTC phase: %{customdata[1]}<extra>%{fullData.name}</extra>'))
        theme(fig,450,title='Spatial evidence · one cell/date/phase marker',xaxis_title='Longitude',yaxis_title='Latitude',legend=dict(orientation='h',y=-.21,x=0,font_size=10))
        fig.update_yaxes(scaleanchor='x',scaleratio=1)
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Cell centers mark the analysis bins; boundary cells may plot outside the detailed polygon while containing detections inside it.')
    with detailcol:
        st.markdown(f'#### {dd} — what the records show')
        daystate=status[status['UTC date']==dd].iloc[0]
        st.metric('Positive groups inside footprint',str(len(selection)))
        st.markdown('**Statewide satellite record context**')
        st.write(f'MODIS raw detections: **{int(daystate["MODIS hotspot records"]):,}**')
        st.write(f'VIIRS raw detections: **{int(daystate["VIIRS hotspot records"]):,}**')
        st.write('**NASA S-NPP anomaly window:** '+('Flagged' if bool(daystate['S-NPP anomaly window']) else 'Not flagged'))
        st.write('**FIRMS missing-data date:** '+('Listed' if bool(daystate['FIRMS listed VIIRS missing date']) else 'Not listed'))
        st.info('A one-sided positive detection is evidence for a reported hotspot, not evidence that the other sensor looked and saw no fire.')
        if len(selection):
            st.dataframe(selection[['Cell ID','Day/night','MODIS records','VIIRS records','At least one time-matched pair','Evidence classification']],hide_index=True,use_container_width=True,height=240)
    with st.expander('Export the complete 316-group audit'):
        st.download_button('Download real V8 Park evidence CSV',groups.to_csv(index=False).encode(),file_name='firelens_v9_park_evidence_316_groups.csv',mime='text/csv')
        st.write('Scope: July 15–31, 2024, exact unbuffered retrospective final Park Fire footprint, 5 km EPSG:6933 grid, UTC date & day/night, 3-hour time tolerance.')
        st.write('The archived geometry was user-exported in the prior V8 workflow and has not been independently re-downloaded and compared with the live CAL FIRE service in this package.')
    st.button("Next: evaluate the experimental model",key="v13_park_to_model",
              type="primary",on_click=_jump_to_california,
              args=("03  REPLAY THE MODEL",))
    source_footer()

if section == '03  REPLAY THE MODEL':
    section_eyebrow('Anticipate / Frozen experimental model')
    st.header('Can seven days of hotspots improve on tomorrow = today?')
    paragraph('This is a retrospective 2025 replay from the saved ConvLSTM V2 checkpoint. The score is an uncalibrated prediction of next-day positive NASA hotspot records in a 10 km grid—not the probability of ignition, spread, or observed absence of fire.')
    # Judge-facing matched-day comparison: identical dates for all four
    # methods; figures come from the separately reproducible frozen-audit script.
    audit=json.loads((ROOT/'v13'/'forecast_audit_results.json').read_text())
    results=audit['methods']
    main_model=results['ConvLSTM']
    previous=results['One-day persistence']
    three_of_seven=results['At least three of seven days']
    union_seven=results['Seven-day union']
    lo,hi=previous['ci95_model_minus_baseline']
    z1,z2,z3,z4=st.columns(4)
    z1.metric('ConvLSTM · matched-day F1',f"{main_model['f1']:.4f}")
    z2.metric('Yesterday = today · F1',f"{previous['f1']:.4f}")
    z3.metric('Active ≥3 of 7 days · F1',f"{three_of_seven['f1']:.4f}")
    z4.metric('Gain vs yesterday',f"+{previous['model_minus_baseline_f1']:.3f}",
              help='Exploratory F1 gain over one-day persistence on the same archived test dates')
    st.caption(f'95% block-bootstrap CI for gain vs yesterday: +{lo:.3f} to +{hi:.3f} F1.')
    table_rows=[]
    for label,record in results.items():
        table_rows.append({
            'Method':label,'F1':round(record['f1'],4),
            '95% block CI · low':round(record['ci95_f1'][0],4),
            '95% block CI · high':round(record['ci95_f1'][1],4),
            'Matched positive cells':record['tp'],
            'Predicted-only cells':record['fp'],
            'Recorded-only cells':record['fn'],
        })
    with st.expander('Full model and baseline evaluation · identical 285 days'):
        st.dataframe(pd.DataFrame(table_rows),hide_index=True,use_container_width=True)
        st.caption(
            'All methods evaluated over March 22–December 31, 2025 (285 UTC days). '
            'The first seven test dates were removed so each 7-day rule has '
            'complete historical labels. A fixed 14-day circular block bootstrap '
            'with 3,000 draws estimates day-clustered uncertainty; its intervals '
            'do not correct for previously inspecting the test period.'
        )
        st.download_button('Download auditable model evaluation (JSON)',
                           json.dumps(audit,indent=2),
                           file_name='firelens_model_research_audit.json',
                           mime='application/json')
    st.caption(
        'The archived test period was previously inspected in model development, '
        'so these are **exploratory** next-day recorded-hotspot-grid overlap scores, '
        'not independent wildfire-forecast accuracy. Baselines use historical '
        'positive reports only; the model threshold was locked on validation.'
    )
    st.divider()
    replay_date=st.select_slider('Choose next-day target date (UTC)',dates,value='2025-09-03')
    idx=dates.index(replay_date)
    predicted=scores[idx]>=threshold
    observed=actual[idx]
    day=counts(predicted,observed)
    cc1,cc2,cc3,cc4=st.columns(4)
    cc1.metric('Matched occupied cells',str(day['tp']))
    cc2.metric('Predicted only',str(day['fp']))
    cc3.metric('Recorded only',str(day['fn']))
    cc4.metric('Day-specific F1',f'{day["f1"]:.3f}')
    if idx>0:
        previous=actual[idx-1]
        pbase=counts(previous,observed)
        st.caption(f'This date: previous-day persistence F1 {pbase["f1"]:.3f}; ConvLSTM F1 {day["f1"]:.3f}. Daily fluctuations may differ substantially from pooled performance.')
    else:
        previous=None
        st.caption('Previous-day raster was not included for the first test date in the compact replay export; see full-period persistence metrics above.')

    # Recreate the same fixed EPSG:6933 grid origin used by the training
    # dataset builder. Overlay a genuine (coarse) California outline so judges
    # can understand the geographic footprint of the 99x103 raster.
    cell_m=int(dcard['grid_km'])*1000
    lo0,la0,lo1,la1=dcard['fixed_bbox_lonlat']
    to_m=Transformer.from_crs('EPSG:4326','EPSG:6933',always_xy=True)
    from_m=Transformer.from_crs('EPSG:6933','EPSG:4326',always_xy=True)
    xx0,yy0=to_m.transform(lo0,la0);xx1,yy1=to_m.transform(lo1,la1)
    xmin=math.floor(min(xx0,xx1)/cell_m)
    ymin=math.floor(min(yy0,yy1)/cell_m)
    row_grid,col_grid=np.indices(observed.shape)
    lon_grid,lat_grid=from_m.transform(
        (xmin+col_grid+.5)*cell_m,(ymin+row_grid+.5)*cell_m
    )
    geo_hover=np.stack([lon_grid,lat_grid],axis=-1)

    figs=make_subplots(
        rows=1,cols=3,horizontal_spacing=.08,
        subplot_titles=['Locked model score','Next-day NASA positive records',
                        'Model vs archived positives']
    )
    figs.add_trace(go.Heatmap(
        z=scores[idx],customdata=geo_hover,
        colorscale=[[0,'#0F2030'],[.35,'#1E6274'],[.67,'#F6B76D'],[1,'#F45B53']],
        zmin=0,zmax=1,showscale=True,
        colorbar=dict(title='Score',len=.54,thickness=9,
                      x=.285,xanchor='right',y=.5),
        hovertemplate='Approx. %{customdata[1]:.2f}°N, %{customdata[0]:.2f}°E'
                      '<br>Uncalibrated model score %{z:.3f}<extra></extra>'
    ),row=1,col=1)
    figs.add_trace(go.Heatmap(
        z=observed.astype(int),customdata=geo_hover,
        zmin=0,zmax=1,
        colorscale=[[0,BG],[.5,BG],[1,TEAL]],showscale=False,
        hovertemplate='Approx. %{customdata[1]:.2f}°N, %{customdata[0]:.2f}°E'
                      '<br>NASA positive record %{z}<extra></extra>'
    ),row=1,col=2)
    confusion=np.zeros(observed.shape,dtype=np.uint8)
    confusion[predicted&observed]=1
    confusion[predicted&~observed]=2
    confusion[~predicted&observed]=3
    discrete=[[0,BG],[.249,BG],[.25,TEAL],[.499,TEAL],
              [.5,ORANGE],[.749,ORANGE],[.75,RED],[1,RED]]
    figs.add_trace(go.Heatmap(
        z=confusion,customdata=geo_hover,zmin=0,zmax=3,
        colorscale=discrete,showscale=True,
        colorbar=dict(title='Record overlap',thickness=12,len=.6,
                      x=1.02,tickvals=[0,1,2,3],
                      ticktext=['No positive','Both','Model only','Recorded only']),
        hovertemplate='Approx. %{customdata[1]:.2f}°N, %{customdata[0]:.2f}°E'
                      '<br>Record-overlap class %{z}<extra></extra>'
    ),row=1,col=3)

    outline_path=ROOT/'v13'/'california_outline.geojson'
    outline=json.loads(outline_path.read_text())
    perimeter=outline['geometry']['coordinates'][0]
    olon=np.array([p[0] for p in perimeter],dtype=float)
    olat=np.array([p[1] for p in perimeter],dtype=float)
    ox,oy=to_m.transform(olon,olat)
    cx=np.asarray(ox)/cell_m-xmin-.5
    cy=np.asarray(oy)/cell_m-ymin-.5
    for col in (1,2,3):
        figs.add_trace(go.Scatter(
            x=cx,y=cy,mode='lines',showlegend=False,hoverinfo='skip',
            line=dict(color='rgba(242,248,252,.90)',width=1.6),
            name='California outline'
        ),row=1,col=col)
    figs.update_xaxes(title_text='10 km grid column → E',showgrid=False,
                      zeroline=False,range=[-.5,observed.shape[1]-.5],
                      tickvals=[0,25,50,75,100])
    figs.update_yaxes(title_text='10 km grid row → N',showgrid=False,
                      zeroline=False,range=[-.5,observed.shape[0]-.5],
                      tickvals=[0,25,50,75])
    theme(figs,485,title=f'California · archived next-day hotspot comparison · {replay_date}',
          showlegend=False)
    figs.update_layout(margin=dict(l=40,r=120,t=85,b=65))
    st.plotly_chart(figs,use_container_width=True)
    st.caption(
        'California outline overlaid in the exact training-grid projection; '
        'hover for approximate cell-center coordinates. White border is a '
        'geographic reference, not a fire perimeter. The three panels show '
        'historical records and a locked experimental model.'
    )

    with st.expander('Download this date’s positive-record and predicted-positive grid cells'):
        idxs=np.argwhere(predicted|observed)
        cell_m=int(dcard['grid_km'])*1000
        lo0,la0,lo1,la1=dcard['fixed_bbox_lonlat']
        proj=Transformer.from_crs('EPSG:4326','EPSG:6933',always_xy=True)
        xlo,ylo=proj.transform(lo0,la0);xhi,yhi=proj.transform(lo1,la1)
        xmin=math.floor(min(xlo,xhi)/cell_m);ymin=math.floor(min(ylo,yhi)/cell_m)
        back=Transformer.from_crs('EPSG:6933','EPSG:4326',always_xy=True)
        lon,lat=back.transform((xmin+idxs[:,1]+.5)*cell_m,(ymin+idxs[:,0]+.5)*cell_m)
        export=pd.DataFrame({'date_utc':replay_date,'grid_row':idxs[:,0],'grid_column':idxs[:,1],
                             'cell_center_lat':lat,'cell_center_lon':lon,
                             'proxy_score':scores[idx][idxs[:,0],idxs[:,1]],
                             'above_frozen_threshold':predicted[idxs[:,0],idxs[:,1]],
                             'recorded_positive':observed[idxs[:,0],idxs[:,1]]})
        st.download_button('Download selected-day positive-union cell audit',export.to_csv(index=False),file_name=f'FireLens_California_replay_{replay_date}.csv',mime='text/csv')
    st.divider()
    with st.expander('Training curve and validation-only model selection'):
        hist=pd.read_csv(PRED/'training_history.csv')
        figure=go.Figure()
        figure.add_trace(go.Scatter(x=hist.epoch,y=hist.train_loss,mode='lines+markers',name='Training loss',line=dict(color=ORANGE)))
        figure.add_trace(go.Scatter(x=hist.epoch,y=hist.val_loss,mode='lines+markers',name='Validation loss',line=dict(color=TEAL)))
        theme(figure,320,title='20 epochs · positive-class weight 10 · best validation-loss epoch 20',xaxis_title='Epoch',yaxis_title='Weighted BCE loss')
        st.plotly_chart(figure,use_container_width=True)
        st.caption('The test period was previously inspected. Full methods and limits are in Methods & Sources.')
    source_footer()

if section == '04  AMAZON 2022–2025':
    from v10.amazon_section import render_amazon
    render_amazon()

if section == 'METHODS & SOURCES':
    section_eyebrow('Validity / Reproducible evidence')
    st.header('What is measured—and what is not')
    st.markdown('''
**Observed:** A FIRMS hotspot report at a reported geolocation and acquisition time. **Not observed here:** whether a satellite imaged every no-record pixel, and whether an unreported pixel was truly fire-free.

**Common cell-day unit:** For any chosen MODIS + VIIRS pair, FireLens can project NASA type-0 presumed vegetation-fire records using the original EPSG:6933 transformer into 10-km cells (configurable). One cell/date is active per sensor if it contains at least one positive record; optional day/night splitting and supplied, documented UTC outage flags preserve context. The resulting table has cell_id, date, modis_active, viirs_active and outage_flag. This is a positive-record union, so unlisted grid days are not verified negatives. This utility is not yet a calibrated global time series.\n\n**Harmonization method:** Bin geolocations into equal-area EPSG:6933 spatial cells; compare matched UTC dates and day/night phases; require acquisition-time compatibility within three hours for the Park Fire audit. No sensor-physics calibration, cloud correction, view-footprint inference, or false-negative sensitivity estimate is claimed.

**Historical geometry:** The Park Fire boundary is the **retrospective final perimeter**, approximately 429,603 acres from the earlier CAL FIRE archive export. It must not be portrayed as the extent of fire on each July 2024 date.

**Experimental ML:** ConvLSTM V2 (one recurrent convolution block, eight hidden channels) uses seven days of 10 km grids with four channels. It forecasts the next day's union of MODIS and VIIRS positive-record grid cells. Train 2022-01-08 to 2024-05-26 (870 sequences); validate 2024-05-27 to 2025-03-14 (292); exploratory test 2025-03-15 to 2025-12-31 (292). Validation selection used threshold 0.675 after 20 epochs and positive-class loss weight 10.

**Important:** Four-year model features include a VIIRS daily-record-presence indicator and documented 2024 S-NPP anomaly windows. These are coarse date-level proxies and cannot verify valid satellite observations at a pixel.
    ''')
    st.subheader('Data lineage')
    st.code('''California July 2024 source observations (kept unchanged):
MODIS  fire_archive_M-C61_850130.csv   4,851 records
VIIRS  fire_archive_SV-C2_850131.csv   7,770 records

Expanded model archive recorded in dataset_card.json:
MODIS  fire_archive_M-C61_850338.csv  33,738 records
VIIRS  fire_archive_SV-C2_850339.csv 139,733 records

Full raw training archives are omitted from the compact deployed application.
Model replay: frozen 2025 test_predictions.npz from trained V2.
Original methods and archived source are preserved in legacy_v8/.''',language='text')
    c1,c2=st.columns(2)
    with c1:
        st.markdown('**Official data and sources**')
        st.markdown('- [NASA FIRMS archives](https://firms.modaps.eosdis.nasa.gov/download/)\n- [NASA FIRMS missing-data page](https://firms.modaps.eosdis.nasa.gov/api/missing_data/)\n- [NASA MODAPS Suomi NPP disruptions](https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html)\n- [CAL FIRE historical perimeters](https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters)')
    with c2:
        st.markdown('**Measured results**')
        st.markdown('- 2024 Park Fire final footprint: 316 positive-only groups, 54 both-positive, 45 time-compatible.\n- Validation F1: ConvLSTM 0.4301; persistence 0.3955.\n- Exploratory test F1: ConvLSTM 0.4212; persistence 0.3818.\n- The model generated 4,363 predicted-only cells vs 5,603 for persistence across the exploratory test period.')
    qa=json.loads((ROOT/'data'/'qa_reproduction.json').read_text())
    st.success('Offline integrity checks passed: original July raw-record totals, the V8 Park evidence denominators, sequence dates and split counts, plus 2025 prediction TP/FP/FN.')
    with st.expander('Reproducibility report (JSON)'):
        st.json(qa)
    st.markdown('**Reproduction:** The package retains the original Park Fire audit and frozen ConvLSTM evaluation arrays. The forecasting view replays archived 2025 scores rather than retraining live.')
    st.markdown(
        "**Sensor identity:** Global country views use MODIS (Terra/Aqua) "
        "and **VIIRS NOAA-20 (JPSS-1)** from the worldwide archive. "
        "California Park Fire and experimental model views use MODIS "
        "and **VIIRS S-NPP**, which is a different satellite."
    )
    st.markdown(
        "**Model verification:** The frozen 285-day audit compares the "
        "ConvLSTM with one-day persistence, seven-day union, and a fixed "
        "three-of-seven-day rule. A 14-day block bootstrap with 3,000 draws "
        "summarizes uncertainty; the test dates were previously inspected."
    )
    st.markdown(
        "**Geographic outline:** Model panels use the true 10-km "
        "EPSG:6933 training-grid origin with a contextual California outline "
        "from [PublicaMundi MappingAPI](https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json). "
        "The outline is not a fire boundary or a validation mask."
    )
    st.caption('The application does not collect personal data, contact fire agencies, or provide operational emergency guidance.')