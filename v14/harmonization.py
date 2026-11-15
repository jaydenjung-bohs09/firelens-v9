"""Common MODIS / VIIRS positive-record cell-day unit.

This function does NOT infer negative observations or valid sensor coverage.
Records must already be a single explicitly chosen VIIRS product for any
cross-sensor comparison. The output includes union-positive cell-days only.
"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

# Reuse the exact original calibration-independent EPSG:6933 projection.
from v10.prepare_region import TRANSFORM, GRID_METERS

COLUMNS=["cell_id","date","modis_active","viirs_active","outage_flag"]


def active_cell_days(
    records: pd.DataFrame,
    *,
    grid_m: int = GRID_METERS,
    separate_daynight: bool = False,
    outage_dates: Iterable[str] = (),
) -> pd.DataFrame:
    """One union-positive grid-cell/day per row, two binary sensor flags.

    Required columns are latitude, longitude, type, sensor_group (or sensor),
    and date (or acq_date). For optional phase, daynight is required and
    the output adds a 'daynight' column. NASA FIRMS type=0 only.

    outage_dates must be externally verified UTC dates; this function never
    invents coverage or outage windows. Even an outage-flagged cell-day
    needs a positive MODIS or VIIRS record to appear in the output.
    """
    if not isinstance(grid_m,int) or grid_m<=0:
        raise ValueError("grid_m must be a positive integer")
    sensor_col="sensor_group" if "sensor_group" in records else "sensor"
    date_col="date" if "date" in records else "acq_date"
    required={"latitude","longitude","type",sensor_col,date_col}
    if separate_daynight:
        required.add("daynight")
    if missing:=required.difference(records.columns):
        raise ValueError(f"Missing required NASA record fields: {sorted(missing)}")
    frame=records.copy()
    frame["type"]=pd.to_numeric(frame["type"],errors="raise")
    frame=frame.loc[frame["type"].eq(0)].copy()
    columns=COLUMNS+(["daynight"] if separate_daynight else [])
    if frame.empty:
        return pd.DataFrame(columns=columns).astype(
            {"modis_active":"int8","viirs_active":"int8","outage_flag":"int8"}
        )
    sensors=frame[sensor_col].astype(str).str.upper()
    sensors=sensors.replace({
        "VIIRS_S_NPP":"VIIRS","VIIRS_NOAA20":"VIIRS",
        "VIIRS NOAA-20":"VIIRS","SNPP":"VIIRS",
    })
    if not sensors.isin(["MODIS","VIIRS"]).all():
        raise ValueError("Sensor values must be MODIS or one specified VIIRS product")
    dates=pd.to_datetime(frame[date_col],errors="raise",utc=True)
    lat=pd.to_numeric(frame["latitude"],errors="raise")
    lon=pd.to_numeric(frame["longitude"],errors="raise")
    if not lat.between(-90,90).all() or not lon.between(-180,180).all():
        raise ValueError("NASA FIRMS coordinates outside valid geographic bounds")
    x,y=TRANSFORM.transform(lon.to_numpy(),lat.to_numpy())
    gx=np.floor(np.asarray(x)/grid_m).astype(np.int64)
    gy=np.floor(np.asarray(y)/grid_m).astype(np.int64)
    frame["cell_id"]=pd.Series(gx,index=frame.index).astype(str)+"_"+pd.Series(gy,index=frame.index).astype(str)
    frame["date"]=dates.dt.strftime("%Y-%m-%d")
    frame["_sensor"]=sensors
    keys=["cell_id","date"]
    if separate_daynight:
        frame["daynight"]=frame["daynight"].astype(str).str.upper()
        if not frame["daynight"].isin(["D","N"]).all():
            raise ValueError("daynight must contain D or N")
        keys.append("daynight")
    distinct=frame[keys+["_sensor"]].drop_duplicates()
    # A pivot marks at least one type-0 record of each product with a 1.
    table=distinct.assign(active=np.int8(1)).pivot_table(
        index=keys,columns="_sensor",values="active",aggfunc="max",
        fill_value=0
    ).reset_index()
    table.columns.name=None
    table["modis_active"]=table["MODIS"] if "MODIS" in table else 0
    table["viirs_active"]=table["VIIRS"] if "VIIRS" in table else 0
    flagged=set(pd.to_datetime(list(outage_dates),utc=True).strftime("%Y-%m-%d"))
    table["outage_flag"]=table["date"].isin(flagged).astype(np.int8)
    for col in ("modis_active","viirs_active","outage_flag"):
        table[col]=table[col].astype(np.int8)
    return table[keys+["modis_active","viirs_active","outage_flag"]].sort_values(
        keys,kind="stable"
    ).reset_index(drop=True)
