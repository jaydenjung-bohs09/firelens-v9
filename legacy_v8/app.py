import io
import numpy as np
import pandas as pd
import plotly.express as px
from pyproj import Transformer
import streamlit as st
from availability import audit_availability, EVENTS, FIRMS_MISSING_SOURCE
from temporal import temporal_agreement, time_window_sensitivity, daily_matching
from coverage_v8 import evidence_groups, summarize_evidence, daily_record_status
from perimeter import fetch_official_park_perimeter, read_perimeter_geojson, filter_perimeter_points, perimeter_deck, perimeter_geojson_bytes
from analysis import (normalize_firms, sensitivity as scientific_sensitivity, overlap as scientific_overlap,
                      park_region, daily_series, quality_table, PARK_START_UTC, PARK_LAT, PARK_LON)

st.set_page_config(page_title='FireLens | NASA Space Apps 2026', page_icon='🔥', layout='wide')
st.title('FireLens')
st.caption('Harmonizing MODIS and VIIRS active-fire observations | NASA Space Apps 2026')
st.info('Research prototype, not an operational fire-warning system. Hotspot detections are not fire perimeters, fire counts, or independent confirmations.')

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def normalize(frame, sensor):
    d = frame.copy()
    required = ['latitude','longitude','acq_date']
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f'{sensor}: missing required columns: {", ".join(missing)}')
    d['latitude'] = pd.to_numeric(d['latitude'], errors='coerce')
    d['longitude'] = pd.to_numeric(d['longitude'], errors='coerce')
    d['date'] = pd.to_datetime(d['acq_date'], errors='coerce').dt.normalize()
    d['sensor_group'] = sensor
    d['frp'] = pd.to_numeric(d['frp'], errors='coerce') if 'frp' in d else np.nan
    d = d.dropna(subset=['date','latitude','longitude'])
    d = d[d.latitude.between(-90,90) & d.longitude.between(-180,180)]
    return d[['date','latitude','longitude','sensor_group','frp']]

def grid_and_calendar(df, cell_km, scope=None):
    """Bin centroids in EPSG:6933, a global equal-area projection (metres).

    This removes latitude-dependent area distortions from the prototype's
    original degree-based grid. Cells are equal-area projected squares,
    not equal ground-distance squares everywhere on Earth.
    """
    d = df.copy()
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
    x, y = transformer.transform(d.longitude.to_numpy(), d.latitude.to_numpy())
    cell_m = cell_km * 1000
    d['grid_x'] = np.floor(np.asarray(x) / cell_m).astype('int64')
    d['grid_y'] = np.floor(np.asarray(y) / cell_m).astype('int64')
    d['cell_id'] = d['grid_x'].astype(str) + '_' + d['grid_y'].astype(str)
    events = d.drop_duplicates(['date','cell_id','sensor_group'])
    by_sensor = events.groupby(['date','sensor_group'])['cell_id'].nunique().unstack(fill_value=0)
    for sensor in ['MODIS', 'VIIRS']:
        if sensor not in by_sensor:
            by_sensor[sensor] = 0
    by_sensor = by_sensor[['MODIS','VIIRS']]
    union = d.groupby('date')['cell_id'].nunique().rename('Unique active grid cells')
    timeline = by_sensor.join(union).reset_index()
    return d, events, timeline

with st.sidebar:
    st.header('1 · Data')
    demo = st.toggle('Use SYNTHETIC demonstration data', value=True,
                     help='Illustrative simulated data only; not NASA observations.')
    modis_upload = st.file_uploader('NASA FIRMS MODIS CSV', type='csv', disabled=demo)
    viirs_upload = st.file_uploader('NASA FIRMS VIIRS CSV', type='csv', disabled=demo)
    st.header('2 · Harmonization')
    cell_km = st.slider('Equal-area grid bin width (km)', 1, 25, 5, 1)
    st.caption('EPSG:6933 equal-area projected squares. Grid-cell counts are not burned area.')
    st.header('3 · Reference')
    st.markdown('[NASA FIRMS download](https://firms.modaps.eosdis.nasa.gov/download/)')
    st.markdown('[Official challenge](https://www.spaceappschallenge.org/2026/challenges/harmonization-of-modis-and-viirs-hot-spots/)')

if demo:
    from pathlib import Path
    modis = load_csv(Path(__file__).parent/'sample_modis_SYNTHETIC.csv')
    viirs = load_csv(Path(__file__).parent/'sample_viirs_SYNTHETIC.csv')
    st.warning('SYNTHETIC DEMO: all plotted hotspot locations and dates are simulated and must not be cited as NASA observations.')
else:
    if not modis_upload or not viirs_upload:
        st.warning('Upload BOTH MODIS and VIIRS FIRMS CSV files from a comparable region and date range, or enable the synthetic demo.')
        st.stop()
    modis, viirs = load_csv(modis_upload), load_csv(viirs_upload)

try:
    df = pd.concat([normalize(modis,'MODIS'), normalize(viirs,'VIIRS')], ignore_index=True)
except ValueError as e:
    st.error(str(e)); st.stop()
if df.empty:
    st.error('No valid observations. Check CSV coordinates and acquisition dates.'); st.stop()

min_day, max_day = df.date.min().date(), df.date.max().date()
start_end = st.date_input('Date window', value=(min_day,max_day), min_value=min_day, max_value=max_day)
if len(start_end) != 2: st.stop()
df = df[(df.date.dt.date >= start_end[0]) & (df.date.dt.date <= start_end[1])].copy()
if df.empty: st.warning('No detections in this date window.'); st.stop()

mapped, events, timeline = grid_and_calendar(df, cell_km, None)
if (mapped.latitude.abs() > 80).any(): st.warning('EPSG:6933 is unsuitable for extreme polar latitudes; restrict to non-polar areas.')
c1,c2,c3,c4 = st.columns(4)
c1.metric('Raw detections',f'{len(df):,}')
c2.metric('Occupied cell-days',f'{timeline["Unique active grid cells"].sum():,}')
c3.metric('MODIS cell-days',f'{timeline.MODIS.sum():,}')
c4.metric('VIIRS cell-days',f'{timeline.VIIRS.sum():,}')

t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs(['Hotspot map','Fire activity calendar','Sensor comparison','Methods & exports','Park Fire case study','Scientific audit','Time-aware matching','Perimeter time matching','Coverage-aware evidence'])
with t1:
    st.subheader('Spatial fire detections')
    st.caption('Each point is a reported hotspot centroid; multiple detections may reflect repeat satellite observations.')
    plotdf = mapped.sample(min(10000,len(mapped)), random_state=11) if len(mapped)>10000 else mapped
    st.map(plotdf, latitude='latitude', longitude='longitude', size=250)
    st.caption('Points include MODIS and VIIRS detections. Use the comparison tab for sensor-specific results.')
with t2:
    st.subheader('Harmonized activity through time')
    full_days = pd.date_range(df.date.min(),df.date.max(),freq='D')
    tl = timeline.set_index('date').reindex(full_days,fill_value=0).rename_axis('date').reset_index()
    fig = px.line(tl,x='date',y='Unique active grid cells',markers=True,
                  title='Unique occupied cells/day (union of both sensors)')
    st.plotly_chart(fig,use_container_width=True)
    tl['weekday']=tl.date.dt.day_name().str[:3]
    tl['week_start']=tl.date.dt.to_period('W-SUN').dt.start_time
    heat = tl.pivot(index='weekday',columns='week_start',values='Unique active grid cells').reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    fig2 = px.imshow(heat,aspect='auto',color_continuous_scale='YlOrRd',labels={'color':'Occupied cells/day'},title='Daily fire activity calendar')
    st.plotly_chart(fig2,use_container_width=True)
    st.subheader('Descriptive anomaly screen')
    st.caption('A high activity day is flagged when the occupied-cell count exceeds the sample 90th percentile. This is descriptive only, not a validated anomaly or forecast. Short samples can mislead.')
    q = tl['Unique active grid cells'].quantile(0.90)
    abnormal=tl[(tl['Unique active grid cells']>q) & (tl['Unique active grid cells']>0)]
    st.dataframe(abnormal[['date','Unique active grid cells']],hide_index=True,use_container_width=True)
with t3:
    st.subheader('Detection comparison')
    long = timeline.melt(id_vars=['date'],value_vars=['MODIS','VIIRS'],var_name='Satellite product',value_name='Occupied cells')
    st.plotly_chart(px.line(long,x='date',y='Occupied cells',color='Satellite product',title='Sensor-specific occupied cells/day'),use_container_width=True)
    # DAILY_SENSOR_AGREEMENT_V1
    presence = (events.assign(present=1)
                .pivot_table(index=['date', 'cell_id'], columns='sensor_group',
                             values='present', aggfunc='max', fill_value=0))
    for name in ('MODIS', 'VIIRS'):
        if name not in presence.columns:
            presence[name] = 0
    presence = presence[['MODIS', 'VIIRS']]
    shared = ((presence['MODIS'] == 1) & (presence['VIIRS'] == 1))
    both = int(shared.sum())
    union_count = len(presence)
    modis_count = int(presence['MODIS'].sum())
    viirs_count = int(presence['VIIRS'].sum())

    a, b, c = st.columns(3)
    a.metric('Jaccard similarity', f'{100 * both / union_count:.2f}%' if union_count else 'N/A')
    b.metric('MODIS overlap', f'{100 * both / modis_count:.2f}%' if modis_count else 'N/A')
    c.metric('VIIRS overlap', f'{100 * both / viirs_count:.2f}%' if viirs_count else 'N/A')
    st.caption(f'MODIS-only: {modis_count - both:,} | Both: {both:,} | '
               f'VIIRS-only: {viirs_count - both:,} | Union: {union_count:,} cell-days')

    daily = presence.assign(shared=shared.astype(int)).groupby(level='date').agg(
        MODIS=('MODIS', 'sum'), VIIRS=('VIIRS', 'sum'), shared=('shared', 'sum'))
    daily['union'] = daily.MODIS + daily.VIIRS - daily.shared
    daily['jaccard_percent'] = 100 * daily.shared / daily.union.replace(0, np.nan)
    daily = daily.reset_index()
    st.plotly_chart(px.line(daily, x='date', y='jaccard_percent', markers=True,
                            title='Daily MODIS–VIIRS cell-day overlap (Jaccard %) ',
                            labels={'jaccard_percent': 'Jaccard similarity (%)',
                                    'date': 'Date'}), use_container_width=True)
    st.dataframe(daily, hide_index=True, use_container_width=True)
    st.download_button('Download daily sensor agreement CSV',
                       daily.to_csv(index=False), 'firelens_sensor_agreement.csv', 'text/csv')
    st.caption('A shared cell-day means both sensor groups detected at least one hotspot '
               'in the same grid cell on the same UTC date, not necessarily the same fire '
               'or overpass. These percentages do not measure either sensor’s accuracy.')
    st.warning('Counts are not normalized for observation opportunity, cloud obstruction, detection threshold, or platform availability. Avoid claims of trend significance or increased fire incidence based solely on this chart.')
with t4:
    st.markdown('''**Pipeline:** Parse FIRMS files → validate coordinates and dates → assign observations to an EPSG:6933 equal-area projected grid → deduplicate within each sensor by grid cell/day → take the union across sensors → display calendar and comparisons.\n\n**Limitations:** A hotspot identifies a satellite-detected thermal anomaly, not a confirmed wildfire or burned area. MODIS and VIIRS differ in spatial resolution, sensitivity, overpass time and historical availability. A union of grid cell-days is an exploratory harmonization, **not** physical cross-calibration. The spatial grid uses EPSG:6933 projected squares; this does not solve differences in sensor detection sensitivity, viewing geometry, clouds, or overpass opportunities.\n\n**Data provenance:** When external NASA CSVs are uploaded, this application analyzes the supplied files in memory. It makes no claim that the synthetic demo data come from NASA.''')
    st.download_button('Download normalized observations',mapped.to_csv(index=False),'firelens_normalized.csv','text/csv')
    st.download_button('Download daily calendar',timeline.to_csv(index=False),'firelens_calendar.csv','text/csv')
    st.markdown('NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/ | Official Challenge: https://www.spaceappschallenge.org/2026/challenges/harmonization-of-modis-and-viirs-hot-spots/')


# FIRELENS SCIENTIFIC UPGRADE V3
# Build a separate rich table so the original, reproducible baseline remains unchanged.
rich = pd.concat([normalize_firms(modis, 'MODIS'),normalize_firms(viirs, 'VIIRS')],ignore_index=True)
rich = rich[(rich.date.dt.date>=start_end[0]) & (rich.date.dt.date<=start_end[1])].copy()
all_days = pd.date_range(start_end[0],start_end[1])
dates_by_sensor = {s:set(rich.loc[rich.sensor_group==s,'date']) for s in ['MODIS','VIIRS']}
common_dates=sorted(dates_by_sensor['MODIS'] & dates_by_sensor['VIIRS'])
missing_viirs = [d for d in all_days if d not in dates_by_sensor['VIIRS']]
missing_modis = [d for d in all_days if d not in dates_by_sensor['MODIS']]
if not demo and (missing_viirs or missing_modis):
    st.warning(f'Coverage alert: within the selected date window, the uploaded MODIS file has no detections on {len(missing_modis)} date(s) and VIIRS has none on {len(missing_viirs)} date(s). This does NOT establish a satellite outage or fire absence. Open Scientific audit before interpreting agreement.')

with t5:
    st.header('Park Fire · Historical case study')
    st.caption('Event reference: CAL FIRE ignition location near Chico, 39.7789° N, 121.76168° W; started July 24, 2024 at 2:52 PM local time. Geographic buffers here are a screening region, NOT the official fire perimeter.')
    st.markdown('[Official CAL FIRE Park Fire incident record](https://www.fire.ca.gov/incidents/2024/7/24/park-fire/)')
    if demo:
        st.warning('Synthetic demo observations are NOT Park Fire observations. Switch to the actual July 2024 NASA CSV files to use this case study.')
    else:
        radius = st.slider('Exploratory radius around documented ignition point (km)',10,100,60,5)
        park = park_region(rich, radius)
        july_study = park[(park.date>=pd.Timestamp('2024-07-15')) & (park.date<=pd.Timestamp('2024-07-31'))].copy()
        before=july_study[july_study.date < PARK_START_UTC]
        after=july_study[july_study.date >= PARK_START_UTC]
        st.caption('July 15–23 = pre-ignition reference period; July 24–31 = post-ignition window. Calendar days use FIRMS UTC dates, so July 24 UTC is not identical to local ignition time.')
        p1,p2,p3=st.columns(3)
        p1.metric('Records in study region',f'{len(july_study):,}')
        p2.metric('Before July 24',f'{len(before):,}')
        p3.metric('July 24–31',f'{len(after):,}')
        local_dates_modis=set(july_study.loc[july_study.sensor_group=='MODIS','date'])
        local_dates_viirs=set(july_study.loc[july_study.sensor_group=='VIIRS','date'])
        if not local_dates_viirs.intersection(pd.date_range('2024-07-24','2024-07-28')):
            st.error('This input contains no VIIRS hotspot records near the ignition region on July 24–28. Do not interpret this as a failure to detect the Park Fire: the uploaded statewide VIIRS file also has no records on these dates.')
        frame=daily_series(july_study, cell_km, '2024-07-15', '2024-07-31')
        fig=px.line(frame.melt(id_vars='date',value_vars=['MODIS','VIIRS','Combined'],var_name='Dataset',value_name='Occupied cells'),
                    x='date',y='Occupied cells',color='Dataset',markers=True,
                    title='Exploratory occupied-cell activity near the Park Fire ignition location')
        fig.add_vline(x=pd.Timestamp('2024-07-24').timestamp()*1000,line_dash='dash',line_color='gray')
        st.plotly_chart(fig,use_container_width=True)
        if len(july_study):
            st.map(july_study[['latitude','longitude']].sample(min(5000,len(july_study)),random_state=12),
                   latitude='latitude',longitude='longitude',size=160)
        st.download_button('Download Park-region daily table',frame.to_csv(index=False),
                           'firelens_park_region_daily.csv','text/csv')
        st.warning('Hotspots inside the circular region could belong to other fires or non-fire thermal anomalies. This is NOT a verified Park Fire pixel assignment, burned-area estimate, fire spread map, or ground-truth validation.')


    # FIRELENS_V5_OFFICIAL_PERIMETER
    st.divider()
    st.subheader('Official historical perimeter · spatial validation')
    st.caption('CAL FIRE FRAP historical perimeter geometry. This is a **retrospective final footprint**: '
               'it must NOT be treated as the fire boundary on July 24–31, 2024.')
    st.markdown('[CAL FIRE historical perimeter data](https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters) · '
                '[ArcGIS source layer](https://services3.arcgis.com/CSK2qijkklTsqQyF/ArcGIS/rest/services/CaliforniaFirePerimeters2024/FeatureServer/0)')
    perimeter_file = st.file_uploader('Optional: upload an official Park Fire GeoJSON', type=['geojson','json'], key='park_perimeter_geojson')
    if st.button('Load CAL FIRE historical Park Fire perimeter', key='load_perimeter'):
        st.session_state['perimeter_load_requested'] = True
    perimeter = None
    perimeter_description = None
    try:
        if perimeter_file is not None:
            perimeter, perimeter_description = read_perimeter_geojson(perimeter_file.getvalue())
        elif st.session_state.get('perimeter_load_requested', False):
            perimeter, perimeter_description = fetch_official_park_perimeter()
    except Exception as exc:
        st.error(f'Could not load or verify the official polygon: {exc}')
        st.info('The app has not substituted a made-up boundary. Try the CAL FIRE service again, or upload a verified GeoJSON from the same source.')
    if perimeter is not None:
        st.success(f'Historical perimeter loaded: {perimeter_description}')
        st.warning('This is the completed fire footprint, not a daily fire progression. A detection inside it is not independent proof it belongs to this particular fire.')
        buffer_km=st.select_slider('Perimeter buffer for sensitivity test (km)',options=[0,1,2,5],value=0)
        july_window = rich[(rich.date >= pd.Timestamp('2024-07-15')) & (rich.date <= pd.Timestamp('2024-07-31'))].copy()
        kept, count_info = filter_perimeter_points(july_window, perimeter, buffer_km)
        k1,k2,k3=st.columns(3)
        k1.metric('Detections inside historical footprint',f'{count_info["inside_unbuffered"]:,}')
        k2.metric('Detections in selected study area',f'{len(kept):,}')
        k3.metric('MODIS / VIIRS',f'{(kept.sensor_group=="MODIS").sum()} / {(kept.sensor_group=="VIIRS").sum()}')
        park_poly_daily=daily_series(kept,cell_km,'2024-07-15','2024-07-31')
        st.plotly_chart(px.line(park_poly_daily.melt(id_vars='date',value_vars=['MODIS','VIIRS','Combined'],var_name='Sensor',value_name='Occupied cells'),
                                x='date',y='Occupied cells',color='Sensor',markers=True,
                                title='Daily occupied cells in the eventual Park Fire footprint'),use_container_width=True)
        st.pydeck_chart(perimeter_deck(perimeter, kept), use_container_width=True)
        st.caption('Outline = historical final Park Fire perimeter. Points = detected hotspot centroids that fall within the selected footprint/buffer. The map does not show daily spread.')
        park_stats=scientific_sensitivity(kept)
        st.dataframe(park_stats,hide_index=True,use_container_width=True)
        st.download_button('Download footprint daily activity CSV',park_poly_daily.to_csv(index=False),
                           'firelens_park_perimeter_daily.csv','text/csv',key='park_poly_daily_export')
        st.download_button('Download footprint sensitivity CSV',park_stats.to_csv(index=False),
                           'firelens_park_perimeter_sensitivity.csv','text/csv',key='park_poly_sens_export')
        st.download_button('Download official perimeter GeoJSON',perimeter_geojson_bytes(perimeter,perimeter_description),
                           'park_fire_official_historical_perimeter.geojson','application/geo+json',key='park_poly_geojson_export')
    else:
        st.info('To run this analysis, click “Load CAL FIRE historical Park Fire perimeter” above. This action requires internet access from Colab.')

with t6:
    st.header('Scientific audit · Do not skip')
    st.caption('Separate observable data quality from unobserved satellite coverage. A zero in FIRMS is NOT proof a satellite viewed the cell and found no fire.')
    st.dataframe(quality_table(rich),hide_index=True,use_container_width=True)
    audit=[]
    for day in all_days:
        audit.append({'UTC date':day,'MODIS records':int(((rich.date==day)&(rich.sensor_group=='MODIS')).sum()),
                      'VIIRS records':int(((rich.date==day)&(rich.sensor_group=='VIIRS')).sum())})
    audit_df=pd.DataFrame(audit)
    audit_df['Has MODIS record']=audit_df['MODIS records']>0
    audit_df['Has VIIRS record']=audit_df['VIIRS records']>0
    st.dataframe(audit_df,hide_index=True,use_container_width=True)
    st.download_button('Download coverage audit CSV',audit_df.to_csv(index=False),
                       'firelens_coverage_audit.csv','text/csv')
    # FIRELENS_V4_AVAILABILITY_TIMELINE
    st.subheader('Instrument availability and documented interruption timeline')
    st.caption('The calendar below separates FIRMS hotspot-record presence from NASA-documented Suomi NPP instrument problems. '
               'A blank hotspot day is NOT proof of a cloud-free observation with no fire; event flags do NOT imply full-day downtime.')
    availability = audit_availability(rich, start_end[0], start_end[1])
    event_count = int(availability['S-NPP documented anomaly window'].sum())
    listed_missing = int(availability['FIRMS listed missing-data date'].sum())
    no_viirs = int((availability['VIIRS records']==0).sum())
    a1,a2,a3=st.columns(3)
    a1.metric('Days with no VIIRS hotspot records',no_viirs)
    a2.metric('Dates in documented anomaly windows',event_count)
    a3.metric('FIRMS missing-data dates',listed_missing)
    import plotly.graph_objects as go
    fig_av=go.Figure()
    fig_av.add_trace(go.Bar(x=availability['UTC date'], y=availability['MODIS records'],name='MODIS records',marker_color='#4477aa'))
    fig_av.add_trace(go.Bar(x=availability['UTC date'], y=availability['VIIRS records'],name='VIIRS records',marker_color='#ee9944'))
    for ev in EVENTS:
        lower=max(pd.Timestamp(ev['start']),pd.Timestamp(start_end[0]))
        upper=min(pd.Timestamp(ev['end']),pd.Timestamp(start_end[1]))
        if lower <= upper:
            fig_av.add_vrect(x0=lower-pd.Timedelta(hours=12),x1=upper+pd.Timedelta(hours=12),
                             fillcolor='crimson',opacity=.12,line_width=0,
                             annotation_text=ev['name'], annotation_position='top left')
    fig_av.update_layout(title='Daily hotspot record counts and S-NPP event windows (UTC)',
                         xaxis_title='UTC date',yaxis_title='Hotspot records',barmode='group',
                         legend_title='Input product')
    st.plotly_chart(fig_av,use_container_width=True)
    st.dataframe(availability,hide_index=True,use_container_width=True)
    st.download_button('Download availability and provenance CSV',availability.to_csv(index=False),
                       'firelens_availability_timeline.csv','text/csv')
    st.markdown('**Source records:** [NASA MODAPS S-NPP outage log]('
                'https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html) · '
                '[NASA FIRMS missing-data listing](https://firms.modaps.eosdis.nasa.gov/api/missing_data/) · '
                '[NASA FIRMS processing updates](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html)')
    st.warning('Anomaly windows and FIRMS missing-data dates are not the same thing. '
               'These flags are known historical S-NPP events for July 2024, not real-time availability monitoring. '
               'The app does not estimate complete overpass coverage, cloud-free observation opportunity, or fire-detection accuracy.')
    st.subheader('Comparison excluding documented S-NPP anomaly-window dates')
    st.caption('Exploratory sensitivity analysis: excludes full UTC calendar days intersecting known event windows, '
               'including partially affected days. This conservative filter is NOT a correction for observation opportunity.')
    excluded=set(availability.loc[availability['S-NPP documented anomaly window'],'UTC date'])
    filtered=rich[~rich.date.isin(excluded)]
    results_all=scientific_overlap(rich,cell_km)[0]
    results_filtered=scientific_overlap(filtered,cell_km)[0]
    exclude_compare=pd.DataFrame([
        {'Analysis':'All dates','Included UTC dates':len(availability),'Union cell-days':results_all['union'],
         'Shared cell-days':results_all['both'],'Jaccard (%)':round(results_all['jaccard_pct'],2)},
        {'Analysis':'Exclude known anomaly windows',
         'Included UTC dates':len(availability)-len(excluded),
         'Union cell-days':results_filtered['union'],
         'Shared cell-days':results_filtered['both'],
         'Jaccard (%)':round(results_filtered['jaccard_pct'],2)}
    ])
    st.dataframe(exclude_compare,hide_index=True,use_container_width=True)
    st.download_button('Download instrument-event sensitivity CSV',
                       exclude_compare.to_csv(index=False),'firelens_event_sensitivity.csv','text/csv')

    st.subheader('Fairer comparison: restrict to dates represented in both input files')
    st.caption('This is a sensitivity check only, not full normalization for overpass opportunity. Days with zero detections may still have valid coverage, so the restriction introduces selection bias.')
    if len(common_dates)==0:
        st.error('No shared detection dates in the current selection.')
    else:
        st.write(f'Dates with >=1 hotspot record from both products: **{len(common_dates)} / {len(all_days)}**')
        only_common=rich[rich.date.isin(common_dates)]
        base=scientific_sensitivity(rich)
        adjusted=scientific_sensitivity(only_common)
        comparison=pd.concat([base.assign(Comparison='All records'),
                              adjusted.assign(Comparison='Shared detection dates')],ignore_index=True)
        st.dataframe(comparison,hide_index=True,use_container_width=True)
        st.plotly_chart(px.line(comparison,x='Grid size (km)',y='Jaccard (%)',color='Comparison',markers=True,
                                title='Sensitivity to date coverage'),use_container_width=True)
        st.download_button('Download coverage-adjusted comparison CSV',comparison.to_csv(index=False),
                           'firelens_coverage_comparison.csv','text/csv')
    st.subheader('Observation and confidence sensitivity')
    st.caption('Day/night groups avoid mixing visibly different acquisition circumstances, but do not match actual satellite overpasses.')
    split=[]
    for phase in ['D','N']:
        d=rich[rich.daynight==phase]
        result=scientific_over=scientific_overlap(d,cell_km)[0]
        split.append({'Day/Night': 'Day' if phase=='D' else 'Night','Raw detections':len(d),
                      'Jaccard (%)':round(result['jaccard_pct'],2), 'Shared cell-days':result['both'],
                      'Union cell-days':result['union']})
    st.dataframe(pd.DataFrame(split),hide_index=True,use_container_width=True)
    st.write('Confidence values use different algorithms and scales for MODIS and VIIRS; they are **not interchangeable**.')
    conf=rich.groupby(['sensor_group','confidence']).size().rename('detections').reset_index()
    st.dataframe(conf,hide_index=True,use_container_width=True)
    st.subheader('UTC acquisition time distributions')
    timed=rich.dropna(subset=['utc_minute']).copy()
    if len(timed):
        timed['UTC hour']=(timed.utc_minute//60).astype(int)
        hist=timed.groupby(['sensor_group','UTC hour']).size().rename('Detections').reset_index()
        st.plotly_chart(px.bar(hist,x='UTC hour',y='Detections',color='sensor_group',barmode='group',
                               title='Reported hotspot detection time (UTC) — NOT a complete overpass schedule'),
                        use_container_width=True)
    st.warning('A scientifically valid observation-opportunity correction requires complete satellite footprints/coverage, including passes with zero hotspots; FIRMS hotspot CSVs alone do not provide those denominators.')
    st.markdown('[NASA FIRMS VIIRS attribute definitions](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)')


# FIRELENS_V6_TIME_AWARE_MATCHING
with t7:
    st.header('Time-aware MODIS–VIIRS detection matching')
    st.caption('Research method: co-located hotspot CENTROIDS in the same projected grid cell, UTC date, '
               'and day/night class. Matches also require acquisition times within the selected window. '
               'This is NOT a full observation-opportunity correction or independent fire identification.')
    h1,h2,h3=st.columns(3)
    with h1:
        match_window=st.select_slider('Matching tolerance (hours)', options=[1,3,6],value=3)
    with h2:
        matching_grid=st.select_slider('Matching grid (km)',options=[2,5,10],value=5)
    with h3:
        omit_events=st.checkbox('Exclude dates in documented S-NPP anomaly windows',value=False,
            help='Excludes entire UTC dates, including partially affected days; a sensitivity check, NOT satellite coverage calibration.')
    study=st.radio('Study area for matching', options=['California dataset','Park Fire 60 km screening circle'],
                   horizontal=True, help='The circle is exploratory, not an official fire boundary. The official final footprint remains in the Park Fire tab.')
    subset=park_region(rich,60) if study.startswith('Park') else rich
    with st.spinner('Matching hotspot centroids by cell and acquisition time...'):
        stats,cell_table,gaps=temporal_agreement(subset,matching_grid,match_window,omit_events)
    a,b,c,d=st.columns(4)
    a.metric('Same-cell / same-phase record groups',f'{stats["co_observed_cell_day_phases"]:,}')
    b.metric('Groups with time-matched pair',f'{stats["temporally_matched_cell_day_phases"]:,}')
    c.metric('Time-matched share',f'{stats["temporal_overlap_pct"]:.2f}%' if pd.notna(stats['temporal_overlap_pct']) else 'N/A')
    d.metric('Matched pairs (1-to-1)',f'{stats["paired_detections"]:,}')
    st.caption('Denominator = cell-date-day/night groups with at least one MODIS and at least one VIIRS hotspot record. '
               'This is conditional on hotspot presence in both files, not conditional on both satellites actually observing the area. '
               'One-to-one pairs are detection records, not fire events.')
    st.subheader('Time-window sensitivity')
    windows=time_window_sensitivity(subset,matching_grid,(1,3,6),omit_events)
    st.dataframe(windows,hide_index=True,use_container_width=True)
    st.plotly_chart(px.line(windows,x='Window (hours)',y='Matched / both-record cell-day-phases (%)',
                            markers=True,title='Sensitivity of time-aware detection matching to tolerance window'),
                    use_container_width=True)
    st.subheader('Daytime versus nighttime results')
    phase_rows=[]
    for phase,title in [('D','Day'),('N','Night')]:
        ps,_,_=temporal_agreement(subset[subset.daynight==phase],matching_grid,match_window,omit_events)
        phase_rows.append({'Phase':title, 'Eligible hotspot records':ps['eligible_records'],
                           'Both-record cell-day-phases':ps['co_observed_cell_day_phases'],
                           'Time-matched cell-day-phases':ps['temporally_matched_cell_day_phases'],
                           'Match fraction (%)':round(ps['temporal_overlap_pct'],2),
                           'Matched pairs':ps['paired_detections']})
    phase_df=pd.DataFrame(phase_rows)
    st.dataframe(phase_df,hide_index=True,use_container_width=True)
    daily_match=daily_matching(cell_table)
    if len(daily_match):
        st.plotly_chart(px.line(daily_match,x='UTC date',y='Match fraction (%)',color='Day/night',
                                markers=True,title='Time-aware matching by UTC day and day/night class'),
                        use_container_width=True)
    if len(gaps):
        st.plotly_chart(px.histogram(gaps,x='Absolute acquisition-time difference (minutes)',nbins=25,
                                     title='Acquisition-time differences for one-to-one matching pairs'),
                        use_container_width=True)
    st.download_button('Download V6 time-window sensitivity CSV',windows.to_csv(index=False),
                       'firelens_v6_time_sensitivity.csv','text/csv',key='v6_window_export')
    st.download_button('Download V6 day/night results CSV',phase_df.to_csv(index=False),
                       'firelens_v6_daynight.csv','text/csv',key='v6_phase_export')
    st.download_button('Download V6 daily matched-cell table CSV',daily_match.to_csv(index=False),
                       'firelens_v6_daily_matching.csv','text/csv',key='v6_daily_export')
    st.download_button('Download V6 per-cell match audit CSV',cell_table.to_csv(index=False),
                       'firelens_v6_cell_audit.csv','text/csv',key='v6_cell_export')
    st.warning('Time-aware matching is intentionally strict: matches must share the same UTC date, '
               'same day/night category and same equal-area projected grid cell. It will miss legitimate '
               'cross-midnight observations and can join unrelated nearby hotspots. '
               'The original same-day Jaccard statistic is a different denominator and should NOT be '
               'compared numerically to this conditional time-match percentage as if both measured accuracy. '
               'Instrument anomalies, clouds, orbit/overpass opportunity, sensor thresholds and geolocation '
               'differences remain unresolved. Neither metric establishes detection recall or precision.')
    st.markdown('[NASA FIRMS input products](https://firms.modaps.eosdis.nasa.gov/content/active_fire/) · '
                '[NASA Suomi NPP outage chronology](https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html)')


# FIRELENS_V7_PERIMETER_TIME_MATCHING
with t8:
    st.header('Perimeter-constrained time-aware harmonization')
    st.caption('Retrospective study using the final 2024 CAL FIRE Park Fire footprint. '
               'The footprint does NOT describe the fire boundary on each date in July. '
               'Hotspot centroid containment is not independent proof of fire identity.')
    st.markdown('[CAL FIRE historical perimeters](https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters) · '
                '[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/active_fire/)')
    geojson_v7 = st.file_uploader('Optional: upload verified CAL FIRE Park Fire GeoJSON',
                                   type=['geojson','json'], key='v7_geojson')
    if st.button('Load official Park Fire perimeter for V7', key='v7_fetch'):
        st.session_state['v7_load'] = True
    poly_v7 = None
    poly_desc_v7 = None
    try:
        if geojson_v7 is not None:
            poly_v7, poly_desc_v7 = read_perimeter_geojson(geojson_v7.getvalue())
        elif st.session_state.get('v7_load', False) or st.session_state.get('perimeter_load_requested',False):
            poly_v7, poly_desc_v7 = fetch_official_park_perimeter()
    except Exception as e:
        st.error('CAL FIRE perimeter could not be loaded: ' + str(e))
        st.info('No estimated or invented polygon is substituted. Check network access or upload the validated GeoJSON.')
    if poly_v7 is None:
        st.info('Click Load official Park Fire perimeter for V7. Internet access is needed unless you upload a verified GeoJSON.')
    else:
        st.success('Validated historical geometry: ' + poly_desc_v7)
        st.caption('All controls affect the tables and charts below. Same source data and date window are used for both geographic definitions.')
        p1,p2,p3=st.columns(3)
        with p1:
            v7_grid=st.select_slider('Grid size (km)',options=[2,5,10],value=5,key='v7_grid')
        with p2:
            v7_hours=st.select_slider('Acquisition-time tolerance (hours)',options=[1,3,6],value=3,key='v7_hours')
        with p3:
            v7_exclude=st.checkbox('Exclude documented S-NPP anomaly dates',value=False,key='v7_anomaly',
              help='Excludes entire UTC dates as a sensitivity check, not as an observation-opportunity correction.')
        v7_buffer=st.select_slider('Perimeter buffer (km)',options=[0,1,2,5],value=0,key='v7_buffer')
        v7_start=pd.Timestamp('2024-07-15'); v7_end=pd.Timestamp('2024-07-31')
        v7_dates=rich[(rich.date>=v7_start)&(rich.date<=v7_end)].copy()
        if len(v7_dates)==0:
            st.warning('No records in July 15–31, 2024 within the selected global date window. Set main Date window to July 1–31, 2024.')
        else:
            with st.spinner('Filtering official historical perimeter and evaluating matching...'):
                v7_foot,v7_foot_counts=filter_perimeter_points(v7_dates,poly_v7,v7_buffer)
                v7_circle=park_region(v7_dates,60)
                v7_foot_stats,v7_foot_table,_=temporal_agreement(v7_foot,v7_grid,v7_hours,v7_exclude)
                v7_circle_stats,v7_circle_table,_=temporal_agreement(v7_circle,v7_grid,v7_hours,v7_exclude)
            st.subheader('Retrospective footprint vs. 60 km screening circle')
            v7_rows=[]
            for label,sub,stats in [('Final footprint + buffer',v7_foot,v7_foot_stats),('60 km screening circle',v7_circle,v7_circle_stats)]:
                v7_rows.append({'Study area':label,'Raw candidate detections':len(sub),
                  'Both-record cell-date-phases':stats['co_observed_cell_day_phases'],
                  'Groups with time match':stats['temporally_matched_cell_day_phases'],
                  'Conditional time-match (%)':round(stats['temporal_overlap_pct'],2) if pd.notna(stats['temporal_overlap_pct']) else np.nan,
                  'One-to-one matched record pairs':stats['paired_detections']})
            v7_comparison=pd.DataFrame(v7_rows)
            st.dataframe(v7_comparison,hide_index=True,use_container_width=True)
            st.warning('The two regions contain different hotspot populations. Their conditional matching percentages are not sensor accuracy rates, and are not directly measures of geographic boundary accuracy.')
            st.subheader('Footprint time-window sensitivity')
            v7_windows=time_window_sensitivity(v7_foot,v7_grid,(1,3,6),v7_exclude)
            st.dataframe(v7_windows,hide_index=True,use_container_width=True)
            st.plotly_chart(px.line(v7_windows,x='Window (hours)',y='Matched / both-record cell-day-phases (%)',markers=True,
                title='Historical footprint: conditional matching by time tolerance'),use_container_width=True)
            st.subheader('Day vs. night within the final footprint')
            phase_v7=[]
            for code,name in [('D','Day'),('N','Night')]:
                ps,_,_=temporal_agreement(v7_foot[v7_foot.daynight==code],v7_grid,v7_hours,v7_exclude)
                phase_v7.append({'Phase':name,'Groups with both records':ps['co_observed_cell_day_phases'],
                    'Groups with time match':ps['temporally_matched_cell_day_phases'],
                    'Conditional time-match (%)':round(ps['temporal_overlap_pct'],2) if pd.notna(ps['temporal_overlap_pct']) else np.nan,
                    'Matched record pairs':ps['paired_detections']})
            v7_phases=pd.DataFrame(phase_v7)
            st.dataframe(v7_phases,hide_index=True,use_container_width=True)
            st.subheader('Historical-perimeter buffer sensitivity')
            buf_rows=[]
            for buf in [0,1,2,5]:
                bf,_=filter_perimeter_points(v7_dates,poly_v7,buf)
                bs,_,_=temporal_agreement(bf,v7_grid,v7_hours,v7_exclude)
                buf_rows.append({'Buffer (km)':buf,'Raw candidate detections':len(bf),
                    'Both-record cell-date-phases':bs['co_observed_cell_day_phases'],
                    'Groups with time match':bs['temporally_matched_cell_day_phases'],
                    'Conditional time-match (%)':round(bs['temporal_overlap_pct'],2) if pd.notna(bs['temporal_overlap_pct']) else np.nan,
                    'Matched record pairs':bs['paired_detections']})
            v7_buffers=pd.DataFrame(buf_rows)
            st.dataframe(v7_buffers,hide_index=True,use_container_width=True)
            st.plotly_chart(px.line(v7_buffers,x='Buffer (km)',y='Conditional time-match (%)',markers=True,
                title='How boundary expansion changes the conditional match rate'),use_container_width=True)
            st.subheader('Daily matching: final historical footprint')
            v7_daily=daily_matching(v7_foot_table)
            st.dataframe(v7_daily,hide_index=True,use_container_width=True)
            if not v7_daily.empty:
                st.plotly_chart(px.line(v7_daily,x='UTC date',y='Match fraction (%)',color='Day/night',markers=True,
                    title='Footprint time-matching by date and day/night'),use_container_width=True)
            st.download_button('Download V7 geographic comparison CSV',v7_comparison.to_csv(index=False),
                 'firelens_v7_region_comparison.csv','text/csv',key='v7_region_csv')
            st.download_button('Download V7 buffer sensitivity CSV',v7_buffers.to_csv(index=False),
                 'firelens_v7_buffer_sensitivity.csv','text/csv',key='v7_buf_csv')
            st.download_button('Download V7 time tolerance CSV',v7_windows.to_csv(index=False),
                 'firelens_v7_time_sensitivity.csv','text/csv',key='v7_win_csv')
            st.download_button('Download V7 day-night CSV',v7_phases.to_csv(index=False),
                 'firelens_v7_daynight.csv','text/csv',key='v7_phase_csv')
            st.download_button('Download V7 daily matching CSV',v7_daily.to_csv(index=False),
                 'firelens_v7_footprint_daily.csv','text/csv',key='v7_daily_csv')
            st.download_button('Download V7 per-cell audit CSV',v7_foot_table.to_csv(index=False),
                 'firelens_v7_footprint_cell_audit.csv','text/csv',key='v7_cell_csv')
            st.download_button('Download verified historical footprint GeoJSON',perimeter_geojson_bytes(poly_v7,poly_desc_v7),
                 'park_fire_verified_historical_footprint.geojson','application/geo+json',key='v7_shape_csv')
            st.caption('Scope: July 15–31, 2024, intersected with the global selected date window. '
              'Retrospective final perimeter only. Group-based matching does not establish two sensors simultaneously observed the same ground area. '
              'Cloud masking, scan geometry, geolocation uncertainty, false positives and observation opportunities are not corrected.')


# FIRELENS_V8_COVERAGE_AWARE_EVIDENCE
with t9:
    st.header('Coverage-aware fire hotspot evidence')
    st.caption('Separates what hotspot CSVs actually show from what they cannot show. '
               'Sensor imaging opportunities and valid negative observations are NOT available in these files.')
    st.info('Interpretive rule: a MODIS-only or VIIRS-only hotspot is a **one-sided positive report**, '
            'NOT proof that the other satellite observed and missed the fire. '
            'Anomaly windows are date-level context, NOT a verified pixel-by-pixel outage map.')
    c1,c2,c3,c4=st.columns(4)
    with c1:
        v8_grid=st.select_slider('V8 matching grid (km)',options=[2,5,10],value=5,key='v8_grid')
    with c2:
        v8_hours=st.select_slider('V8 time tolerance (hours)',options=[1,3,6],value=3,key='v8_hours')
    with c3:
        v8_area=st.selectbox('Study geography', ['California dataset','Park Fire 60 km circle','Park Fire final footprint'],key='v8_area')
    with c4:
        v8_dates=st.selectbox('Date scope',['Current selected date window','Park Fire: July 15–31, 2024'],key='v8_dates')
    v8_frame=rich.copy()
    v8_first,v8_last=(pd.Timestamp(start_end[0]),pd.Timestamp(start_end[1]))
    if v8_dates.startswith('Park Fire'):
        v8_first=max(v8_first,pd.Timestamp('2024-07-15'))
        v8_last=min(v8_last,pd.Timestamp('2024-07-31'))
        v8_frame=v8_frame[(v8_frame.date>=v8_first)&(v8_frame.date<=v8_last)].copy()
    if v8_area=='Park Fire 60 km circle':
        v8_frame=park_region(v8_frame,60)
    elif v8_area=='Park Fire final footprint':
        st.caption('Retrospective *eventual* Park Fire footprint; do not interpret as a daily fire boundary.')
        uploaded_v8=st.file_uploader('Optional CAL FIRE Park Fire GeoJSON',type=['geojson','json'],key='v8_boundary')
        use_local=st.checkbox('Use bundled, previously exported PARK 2024 geometry',value=True,
            help='This file was exported by the earlier V7 app and has not been independently cross-checked with the CAL FIRE layer.',key='v8_local')
        v8_poly=None
        try:
            if uploaded_v8 is not None:
                v8_poly,_=read_perimeter_geojson(uploaded_v8.getvalue())
            elif use_local:
                from pathlib import Path
                v8_poly,_=read_perimeter_geojson((Path(__file__).parent/'park_fire_verified_historical_footprint_USER_EXPORT.geojson').read_bytes())
            elif st.button('Download historical perimeter from CAL FIRE',key='v8_getpoly'):
                v8_poly,_=fetch_official_park_perimeter()
        except Exception as exc:
            st.error('Could not validate Park Fire geometry: '+str(exc))
        if v8_poly is None:
            st.warning('Choose a valid PARK 2024 GeoJSON or the bundled V7 export. No approximate polygon is substituted.')
            st.stop()
        v8_frame,_=filter_perimeter_points(v8_frame,v8_poly,0)
    if v8_first>v8_last:
        st.warning('Selected date range does not overlap the Park Fire case-study period.')
    elif v8_frame.empty:
        st.warning('No positive hotspot records within this study selection.')
    else:
        with st.spinner('Building grouped hotspot evidence and coverage flags...'):
            v8_group=evidence_groups(v8_frame,v8_grid,v8_hours)
            v8_sum=summarize_evidence(v8_group)
            v8_daily=daily_record_status(v8_frame,v8_first,v8_last)
        st.subheader('Three different denominators — do not confuse them')
        s1,s2,s3=st.columns(3)
        s1.metric('Both-positive groups / all positive groups',
            f'{v8_sum["both"]:,} / {v8_sum["union"]:,}',
            help='Same-cell same-date same-phase positivity, not verified opportunity overlap.')
        s2.metric('Time-compatible / both-positive groups',
            f'{v8_sum["time_matched"]:,} / {v8_sum["both"]:,}',
            help='This is the V7 conditional result; groups where one satellite has no hotspot are excluded.')
        s3.metric('Time-compatible / all positive groups',
            f'{v8_sum["time_matched"]:,} / {v8_sum["union"]:,}',
            help='Descriptive share among all groups with any hotspot, not accuracy or true overpass-corrected agreement.')
        st.write('**Descriptive percentages (not sensor performance or accuracy):**')
        st.write(f'Both-positive fraction: **{v8_sum["positive_overlap_pct"]:.2f}%** · '
                 f'Conditional time match: **{v8_sum["conditional_temporal_pct"]:.2f}%** · '
                 f'Time match among all positive groups: **{v8_sum["union_time_match_pct"]:.2f}%**')
        status_counts=v8_group['Evidence classification'].value_counts().rename_axis('Evidence category').reset_index(name='Cell-date-phases')
        st.subheader('Evidence classes, without invented negative observations')
        st.dataframe(status_counts,hide_index=True,use_container_width=True)
        st.plotly_chart(px.bar(status_counts,x='Evidence category',y='Cell-date-phases',
            title='Positive-record evidence by class'),use_container_width=True)
        st.caption(f'MODIS-positive-only groups: {v8_sum["modis_only"]:,}; '
                   f'VIIRS-positive-only: {v8_sum["viirs_only"]:,}; '
                   f'both positive but beyond matching tolerance: {v8_sum["both_outside_window"]:,}. '
                   f'Of MODIS-only groups, {v8_sum["flagged_modis_only"]:,} occur on NASA-listed VIIRS missing-data dates. '
                   'Flags do not establish actual imaging coverage at a specific cell.')
        st.subheader('Calendar audit — zero records versus documented anomalies')
        st.dataframe(v8_daily,hide_index=True,use_container_width=True)
        st.plotly_chart(px.line(v8_daily,x='UTC date',y=['MODIS hotspot records','VIIRS hotspot records'],
                        title='Daily positive hotspot records (not observed-fire-free satellite passes)'),use_container_width=True)
        st.subheader('Effect of excluding S-NPP anomaly windows')
        v8_excluded=v8_group.loc[~v8_group['Instrument anomaly window']].copy()
        v8_filtered=summarize_evidence(v8_excluded)
        comparison=pd.DataFrame([
          {'Date handling':'All selected days',**v8_sum},
          {'Date handling':'Exclude documented affected UTC days',**v8_filtered}
        ])
        st.dataframe(comparison,hide_index=True,use_container_width=True)
        st.warning('Removing whole anomaly-window days is a sensitivity check, not a validated coverage adjustment. '
                   'Partial-day observations may be discarded. The positive-only denominator is selection-dependent.')
        st.subheader('Limitations and next validation needed')
        st.markdown('**Not available from these files:** per-cell satellite observation footprints, clear-sky scan opportunities, '
                    'valid negative observations, probability of detection, precise scan-time alignment, or ground-truth fire identity. '
                    'Thus **false-negative rates, precision/recall, or outage-corrected accuracy cannot be calculated**.')
        st.download_button('Download V8 group evidence CSV',v8_group.to_csv(index=False),
            'firelens_v8_evidence_groups.csv','text/csv',key='v8_groups_export')
        st.download_button('Download V8 daily coverage-status CSV',v8_daily.to_csv(index=False),
            'firelens_v8_daily_status.csv','text/csv',key='v8_daily_export')
        st.download_button('Download V8 sensitivity comparison CSV',comparison.to_csv(index=False),
            'firelens_v8_anomaly_sensitivity.csv','text/csv',key='v8_sens_export')
        st.caption('Source notes: NASA FIRMS positive-hotspot archive; NASA MODAPS S-NPP event log; '
                   'NASA FIRMS missing-data dates; retrospective CAL FIRE Park Fire geometry.')
