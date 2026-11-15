"""Coverage-aware hotspot evidence summaries; NOT an imaging-coverage estimator.

A FIRMS hotspot archive includes positive detections, not the satellite's full
image footprint or observed-but-zero-fire grid cells. Thus labels differentiate
co-reported hotspots, a one-sided positive report, and documented anomalies
without implying true disagreement or classifying all unobserved cells.
"""
import numpy as np
import pandas as pd
from analysis import add_grid
from availability import EVENTS, FIRMS_REPORTED_MISSING
from temporal import temporal_agreement

def _known_flags(day):
    d=pd.Timestamp(day).date()
    return (any(e['start']<=d<=e['end'] for e in EVENTS), d in FIRMS_REPORTED_MISSING)

def evidence_groups(frame, grid_km=5, tolerance_hours=3):
    """One row per positive-hotspot cell/date/D-N with explicit evidence label."""
    _, table, _ = temporal_agreement(frame,grid_km,tolerance_hours,False)
    if table.empty:
        return pd.DataFrame(columns=['UTC date','Cell ID','Day/night','MODIS records','VIIRS records',
          'Both sensors have hotspot records','At least one time-matched pair','Matched pairs',
          'Instrument anomaly window','FIRMS listed VIIRS missing-data date','Evidence classification','Observation opportunity'])
    out=table.copy()
    annotations=out['UTC date'].map(_known_flags)
    out['Instrument anomaly window']=annotations.map(lambda x:x[0])
    out['FIRMS listed VIIRS missing-data date']=annotations.map(lambda x:x[1])
    def classify(r):
        m=r['MODIS records']>0; v=r['VIIRS records']>0
        if m and v:
            return 'Both positive; time-compatible' if r['At least one time-matched pair'] else 'Both positive; no pair within tolerance'
        if m:
            return 'MODIS positive; VIIRS missing-data flag' if r['FIRMS listed VIIRS missing-data date'] else 'MODIS positive only; VIIRS opportunity unknown'
        return 'VIIRS positive only; MODIS opportunity unknown'
    out['Evidence classification']=out.apply(classify,axis=1)
    out['Observation opportunity']='Not established by hotspot-only FIRMS files'
    return out

def summarize_evidence(groups):
    if groups.empty:
        return dict(union=0,both=0,time_matched=0,modis_only=0,viirs_only=0,both_outside_window=0,
          positive_overlap_pct=np.nan,conditional_temporal_pct=np.nan,union_time_match_pct=np.nan,
          single_sensor=0,flagged_modis_only=0)
    m=groups['MODIS records']>0
    v=groups['VIIRS records']>0
    both=m&v
    time=groups['At least one time-matched pair'].astype(bool)
    return dict(union=len(groups),both=int(both.sum()),time_matched=int(time.sum()),
      modis_only=int((m&~v).sum()),viirs_only=int((v&~m).sum()),
      both_outside_window=int((both&~time).sum()),
      positive_overlap_pct=100*both.sum()/len(groups),
      conditional_temporal_pct=100*time.sum()/both.sum() if both.any() else np.nan,
      union_time_match_pct=100*time.sum()/len(groups),
      single_sensor=int((m^v).sum()),
      flagged_modis_only=int((m&~v&groups['FIRMS listed VIIRS missing-data date']).sum()))

def daily_record_status(frame,start,end):
    """Every UTC calendar date, including days with no records, without inferring coverage."""
    dates=pd.date_range(start,end,freq='D')
    counts=frame.groupby(['date','sensor_group']).size().unstack(fill_value=0)
    rows=[]
    for dt in dates:
        m=int(counts.loc[dt,'MODIS']) if dt in counts.index and 'MODIS' in counts else 0
        v=int(counts.loc[dt,'VIIRS']) if dt in counts.index and 'VIIRS' in counts else 0
        affected,missing=_known_flags(dt)
        rows.append({'UTC date':dt,'MODIS hotspot records':m,'VIIRS hotspot records':v,
         'S-NPP anomaly window':affected,'FIRMS listed VIIRS missing date':missing,
         'VIIRS record status':('NASA-listed missing-data date' if missing else
             'Documented anomaly window; status may be partial' if affected else
             'Positive records present' if v else 'Zero records; observation opportunity unknown'),
         'Non-detection imaging coverage verified':'No — hotspot records only'})
    return pd.DataFrame(rows)
