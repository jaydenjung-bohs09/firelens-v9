"""FireLens science utilities.  Presence is not fire truth, area or sensor accuracy."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pyproj import Transformer

TRANSFORMER = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
PARK_LAT, PARK_LON = 39.7789, -121.76168
PARK_START_UTC = pd.Timestamp('2024-07-24')

def normalize_firms(frame: pd.DataFrame, sensor: str) -> pd.DataFrame:
    d = frame.copy()
    required = {'latitude', 'longitude', 'acq_date'}
    if not required.issubset(d):
        raise ValueError(f'{sensor}: missing {sorted(required-set(d.columns))}')
    d['latitude'] = pd.to_numeric(d.latitude, errors='coerce')
    d['longitude'] = pd.to_numeric(d.longitude, errors='coerce')
    d['date'] = pd.to_datetime(d.acq_date, errors='coerce').dt.normalize()
    d['sensor_group'] = sensor
    d['frp'] = pd.to_numeric(d['frp'], errors='coerce') if 'frp' in d else np.nan
    d['daynight'] = d['daynight'].astype(str).str.upper().str.strip() if 'daynight' in d else '?'
    d['daynight'] = d['daynight'].where(d.daynight.isin(['D','N']), '?')
    d['confidence'] = d['confidence'].astype(str).str.lower().str.strip() if 'confidence' in d else '?'
    d['type'] = pd.to_numeric(d['type'], errors='coerce') if 'type' in d else np.nan
    d['satellite'] = d['satellite'].astype(str) if 'satellite' in d else '?'
    if 'acq_time' in d:
        mins = pd.to_numeric(d['acq_time'], errors='coerce')
        hour = np.floor(mins / 100)
        minute = mins % 100
        valid = hour.between(0,23) & minute.between(0,59)
        d['utc_minute'] = np.where(valid, hour*60+minute, np.nan)
    else:
        d['utc_minute'] = np.nan
    d = d.dropna(subset=['date','latitude','longitude'])
    d = d[d.latitude.between(-90,90) & d.longitude.between(-180,180)].copy()
    # Local time must NOT be inferred from a UTC acquisition date without conversion.
    d['utc_timestamp'] = d.date + pd.to_timedelta(d.utc_minute, unit='m')
    cols = ['date','latitude','longitude','sensor_group','frp','daynight','confidence',
            'type','satellite','utc_minute','utc_timestamp']
    return d[cols].reset_index(drop=True)

def add_grid(df: pd.DataFrame, cell_km: float) -> pd.DataFrame:
    d = df.copy()
    x,y = TRANSFORMER.transform(d.longitude.to_numpy(),d.latitude.to_numpy())
    unit = float(cell_km)*1000
    d['grid_x'] = np.floor(np.asarray(x)/unit).astype('int64')
    d['grid_y'] = np.floor(np.asarray(y)/unit).astype('int64')
    d['cell_id'] = d.grid_x.astype(str)+'_'+d.grid_y.astype(str)
    return d

def overlap(df: pd.DataFrame, cell_km: float, by_daynight=False) -> tuple[dict,pd.DataFrame]:
    if df.empty:
        return {'modis':0,'viirs':0,'both':0,'union':0,'jaccard_pct':np.nan,'modis_overlap_pct':np.nan,'viirs_overlap_pct':np.nan},pd.DataFrame()
    d=add_grid(df,cell_km)
    keys=['date','cell_id']+(['daynight'] if by_daynight else [])
    # Deduplicate independent FIRMS detections falling in same day/cell (and optionally D/N).
    p=d[keys+['sensor_group']].drop_duplicates()
    p['present']=1
    presence=p.pivot_table(index=keys,columns='sensor_group',values='present',aggfunc='max',fill_value=0)
    for sensor in ['MODIS','VIIRS']:
        if sensor not in presence: presence[sensor]=0
    presence=presence[['MODIS','VIIRS']]
    m=presence.MODIS.astype(bool); v=presence.VIIRS.astype(bool)
    both=int((m&v).sum()); uni=int((m|v).sum()); nm=int(m.sum());nv=int(v.sum())
    stats={'modis':nm,'viirs':nv,'both':both,'union':uni,
           'jaccard_pct':100*both/uni if uni else np.nan,
           'modis_overlap_pct':100*both/nm if nm else np.nan,
           'viirs_overlap_pct':100*both/nv if nv else np.nan}
    return stats,presence.reset_index()

def sensitivity(df:pd.DataFrame, grids=(2,5,10), by_daynight=False) -> pd.DataFrame:
    rows=[]
    for g in grids:
        stats,_=overlap(df,g,by_daynight=by_daynight)
        rows.append({'Grid size (km)':g,'MODIS cell-days':stats['modis'],'VIIRS cell-days':stats['viirs'],
                     'Shared cell-days':stats['both'],'Union cell-days':stats['union'],
                     'Jaccard (%)':round(stats['jaccard_pct'],2)})
    return pd.DataFrame(rows)

def haversine_km(lat,lon,lat0=PARK_LAT,lon0=PARK_LON):
    p1=np.deg2rad(np.asarray(lat,dtype=float));p0=np.deg2rad(lat0)
    dp=p1-p0;dl=np.deg2rad(np.asarray(lon,dtype=float)-lon0)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p0)*np.sin(dl/2)**2
    return 6371.0088*2*np.arcsin(np.minimum(1,np.sqrt(a)))

def park_region(df:pd.DataFrame, radius_km=60) -> pd.DataFrame:
    return df.loc[haversine_km(df.latitude,df.longitude)<=radius_km].copy()

def daily_series(df:pd.DataFrame, cell_km:float, date_start=None,date_end=None) -> pd.DataFrame:
    g=add_grid(df,cell_km) if len(df) else df.copy()
    if len(g):
        p=g.drop_duplicates(['date','cell_id','sensor_group'])
        pivot=p.groupby(['date','sensor_group']).size().unstack(fill_value=0)
        for name in ['MODIS','VIIRS']:
            if name not in pivot: pivot[name]=0
        union=g.groupby('date')['cell_id'].nunique()
        out=pivot[['MODIS','VIIRS']].join(union.rename('Combined')).reset_index()
    else:
        out=pd.DataFrame(columns=['date','MODIS','VIIRS','Combined'])
    if date_start is not None and date_end is not None:
        full=pd.date_range(date_start,date_end)
        out=out.set_index('date').reindex(full,fill_value=0).rename_axis('date').reset_index()
    return out

def quality_table(df:pd.DataFrame) -> pd.DataFrame:
    return df.groupby('sensor_group',as_index=False).agg(
        detections=('date','size'),
        first_date=('date','min'),last_date=('date','max'),
        missing_frp=('frp',lambda s:int(s.isna().sum())),
        missing_utc_time=('utc_minute',lambda s:int(s.isna().sum())),
        unique_observation_dates=('date','nunique'))
