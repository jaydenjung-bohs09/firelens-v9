"""Time-aware but NOT opportunity-corrected MODIS/VIIRS hotspot comparisons.

A match is a pair of reported hotspot centroids in the SAME projected grid
cell, SAME UTC date and SAME D/N classification, within a user-selected time
window. The greedy sorted two-pointer algorithm provides a maximum-cardinality
one-to-one time-window matching for two sorted time sequences, but does not
prove both records correspond to the same fire or same footprint.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from analysis import add_grid
from availability import EVENTS

VALID_PHASES = ('D','N')

def _match_sorted(a, b, threshold_minutes):
    """Maximum-cardinality 1:1 matching in 1D with |time_a-time_b|<=threshold."""
    a=np.sort(np.asarray(a,dtype=float))
    b=np.sort(np.asarray(b,dtype=float))
    i=j=0; gaps=[]
    while i<len(a) and j<len(b):
        delta=a[i]-b[j]
        if abs(delta)<=threshold_minutes:
            gaps.append(float(abs(delta)));i+=1;j+=1
        elif delta < 0:
            i+=1
        else:
            j+=1
    return gaps

def _filter_known_windows(df):
    affected=set()
    for e in EVENTS:
        affected.update(pd.date_range(e['start'],e['end'],freq='D'))
    return df.loc[~df.date.isin(affected)].copy()

def temporal_agreement(frame, grid_km=5, window_hours=3, exclude_anomalies=False):
    """Return headline results, cell-day-D/N table, and matched time differences.

    Cell-day phase overlap counts are sensor-presence summaries, NOT the number
    of independently verified fires, viewable cells or satellite opportunities.
    Pair-level counts can vary with repeated points/passes in a single cell.
    """
    if float(window_hours)<0: raise ValueError('window_hours must be nonnegative')
    df=frame.copy()
    if exclude_anomalies: df=_filter_known_windows(df)
    for c in ['date','latitude','longitude','utc_minute','daynight','sensor_group']:
        if c not in df.columns: raise ValueError(f'Missing column {c}')
    df=df[df.daynight.isin(VALID_PHASES) & df.utc_minute.notna()].copy()
    if df.empty:
        return {'eligible_records':0, 'co_observed_cell_day_phases':0,
                'temporally_matched_cell_day_phases':0, 'temporal_overlap_pct':np.nan,
                'paired_detections':0, 'median_lag_min':np.nan,
                'candidate_modis_records':0, 'candidate_viirs_records':0,
                'cell_day_phases_union':0, 'same_day_phase_jaccard_pct':np.nan},pd.DataFrame(),pd.DataFrame()
    d=add_grid(df,float(grid_km))
    rows=[]; all_gaps=[]
    # Grouping on (date,cell,phase) avoids mistakenly matching day against night.
    for (day,cell,phase), group in d.groupby(['date','cell_id','daynight'],sort=True):
        modis=group.loc[group.sensor_group=='MODIS','utc_minute'].to_numpy(dtype=float)
        viirs=group.loc[group.sensor_group=='VIIRS','utc_minute'].to_numpy(dtype=float)
        gaps=_match_sorted(modis,viirs,float(window_hours)*60) if len(modis) and len(viirs) else []
        rows.append({'UTC date':day,'Cell ID':cell,'Day/night':phase,
                     'MODIS records':len(modis),'VIIRS records':len(viirs),
                     'Both sensors have hotspot records':bool(len(modis) and len(viirs)),
                     'At least one time-matched pair':bool(gaps),
                     'Matched pairs':len(gaps),
                     'Minimum paired time difference (min)':min(gaps) if gaps else np.nan})
        all_gaps.extend(gaps)
    table=pd.DataFrame(rows)
    both=table['Both sensors have hotspot records']
    match=table['At least one time-matched pair']
    m=(table['MODIS records']>0).sum();v=(table['VIIRS records']>0).sum()
    stats={
        'eligible_records':len(d),
        'candidate_modis_records':int((d.sensor_group=='MODIS').sum()),
        'candidate_viirs_records':int((d.sensor_group=='VIIRS').sum()),
        'co_observed_cell_day_phases':int(both.sum()),
        'temporally_matched_cell_day_phases':int(match.sum()),
        'temporal_overlap_pct':100*int(match.sum())/int(both.sum()) if both.any() else np.nan,
        'paired_detections':len(all_gaps),
        'median_lag_min':float(np.median(all_gaps)) if all_gaps else np.nan,
        'cell_day_phases_union':len(table),
        'same_day_phase_jaccard_pct':100*int(both.sum())/len(table) if len(table) else np.nan,
    }
    lags=pd.DataFrame({'Absolute acquisition-time difference (minutes)':all_gaps})
    return stats,table,lags

def time_window_sensitivity(frame,grid_km=5,windows=(1,3,6),exclude_anomalies=False):
    rows=[]
    for hours in windows:
        stats,_,_=temporal_agreement(frame,grid_km,hours,exclude_anomalies)
        rows.append({'Window (hours)':hours,'Grid (km)':grid_km,
                     'Cell-day-phases with records from both':stats['co_observed_cell_day_phases'],
                     'Cell-day-phases with time match':stats['temporally_matched_cell_day_phases'],
                     'Matched / both-record cell-day-phases (%)':round(stats['temporal_overlap_pct'],2),
                     'One-to-one matched pairs':stats['paired_detections'],
                     'Median time difference (minutes)':round(stats['median_lag_min'],1)})
    return pd.DataFrame(rows)

def daily_matching(table):
    if table.empty:
        return pd.DataFrame(columns=['UTC date','Day/night','Both-sensor cell-day-phases','Time-matched cell-day-phases','Match fraction (%)'])
    out=table.groupby(['UTC date','Day/night'],as_index=False).agg(
        **{'Both-sensor cell-day-phases':('Both sensors have hotspot records','sum'),
           'Time-matched cell-day-phases':('At least one time-matched pair','sum')})
    out['Match fraction (%)']=100*out['Time-matched cell-day-phases']/out['Both-sensor cell-day-phases'].replace(0,np.nan)
    return out
