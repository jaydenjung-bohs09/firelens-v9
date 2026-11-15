"""Historical S-NPP event annotations and CSV record-coverage audit.

Caution: hotspot CSVs do not contain complete acquisition opportunities.
Dates are UTC and these event windows are day-level flags, not evidence of
24-hour satellite downtime.
"""
from datetime import date
import pandas as pd

EVENTS = [
    dict(start=date(2024,7,9), end=date(2024,7,16),
         name='S-NPP GPS anomaly',
         note='Geolocation/production disruptions. Date range denotes an affected window, not 24-hour outage on every day.',
         url='https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html'),
    dict(start=date(2024,7,24), end=date(2024,7,29),
         name='S-NPP command-data processor anomaly',
         note='Science data and processing interrupted; some dates have partial-day impact. Does not establish all-day outage for each date.',
         url='https://modaps.modaps.eosdis.nasa.gov/services/production/outages_suomi_npp.html')
]
# Dates listed on NASA FIRMS missing-data page for S-NPP VIIRS (2024)
# These align with the supplied July 2024 archives; do not generalize to all products.
FIRMS_REPORTED_MISSING = {
    date(2024,7,i) for i in [10,11,12,13,14,15,25,26,27,28]
}
FIRMS_MISSING_SOURCE = 'https://firms.modaps.eosdis.nasa.gov/api/missing_data/'

def audit_availability(rich, start, end):
    """Return full calendar with sensor counts and *source-labeled* flags.

    A zero record count is never equated to a valid zero-fire observation.
    """
    days=pd.date_range(start,end,freq='D')
    df=rich.copy()
    df['date']=pd.to_datetime(df['date']).dt.normalize()
    counts=df.groupby(['date','sensor_group']).size().unstack(fill_value=0)
    records=[]
    for day in days:
        d=day.date()
        m=int(counts.loc[day,'MODIS']) if day in counts.index and 'MODIS' in counts.columns else 0
        v=int(counts.loc[day,'VIIRS']) if day in counts.index and 'VIIRS' in counts.columns else 0
        hits=[ev for ev in EVENTS if ev['start']<=d<=ev['end']]
        flagged=bool(hits)
        reported_missing=d in FIRMS_REPORTED_MISSING
        if reported_missing:
            status='NASA FIRMS missing-data date'
        elif flagged:
            status='Documented anomaly window (partial-day status possible)'
        elif v>0:
            status='VIIRS hotspot records present'
        else:
            status='No VIIRS hotspot records; coverage unknown'
        records.append({
            'UTC date':day,
            'MODIS records':m, 'VIIRS records':v,
            'S-NPP documented anomaly window':flagged,
            'FIRMS listed missing-data date':reported_missing,
            'Instrument event':'; '.join(x['name'] for x in hits) if hits else '',
            'VIIRS record availability':status,
            'Evidence URL':FIRMS_MISSING_SOURCE if reported_missing else (hits[0]['url'] if hits else '')
        })
    return pd.DataFrame(records)
