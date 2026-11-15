"""Country month calendar in VIIRS NOAA-20 archived-positive-record units.

2000–2017 are MODIS-modeled NOAA-20 count estimates, NOT observed NOAA-20.
2018+ source counts are observed NOAA-20 archived positives; Jan–Mar 2018
are unavailable, and 2000 MODIS Jan–Oct are unavailable.
80% bounds are empirical 2019–2025 LOYO residual quantiles, NOT
verified coverage for historical years or actual wildfires.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v15.calibration import complete_overlap_years,fit_calibration
from v15.validation import selected_method,prediction_intervals

OUTPUT_COLUMNS=["country","year","month","value","source","lower","upper"]

def country_calendar(rows:pd.DataFrame,validation:pd.DataFrame,
                     heldout:pd.DataFrame,iso3:str)->pd.DataFrame:
    chosen=rows.loc[rows.iso3.eq(iso3)].copy().sort_values(["year","month"])
    if chosen.empty:raise ValueError(f"Country absent from source: {iso3}")
    winner=selected_method(validation)
    model=fit_calibration(rows,years=complete_overlap_years(rows))
    bounds=prediction_intervals(heldout,winner)
    qlo,qhi=bounds.get(str(chosen.continent.iloc[0]),bounds["GLOBAL"])
    historic=chosen.loc[chosen.year.lt(2018)].copy()
    observed=chosen.loc[chosen.year.ge(2018)].copy()
    historic["value"]=model.predict(historic,winner).predicted.to_numpy()
    historic["source"]="modeled"
    historic["lower"]=np.maximum(0,np.expm1(np.log1p(historic.value)+qlo))
    historic["upper"]=np.expm1(np.log1p(historic.value)+qhi)
    observed["source"]="observed"
    observed["value"]=observed.viirs.copy()
    observed.loc[observed.year.eq(2018)&observed.month.lt(4),"value"]=np.nan
    observed["lower"]=np.nan
    observed["upper"]=np.nan
    out=pd.concat([historic,observed],ignore_index=True)
    out["country"]=chosen.country.iloc[0]
    return out[OUTPUT_COLUMNS].sort_values(["year","month"]).reset_index(drop=True)

def monthly_anomalies(calendar:pd.DataFrame)->pd.DataFrame:
    out=calendar.copy()
    baseline=out.groupby("month")["value"].median()
    out["historical_median"]=out.month.map(baseline)
    out["anomaly_pct"]=np.where(
        out.historical_median.gt(0)&out.value.notna(),
        100*(out.value-out.historical_median)/out.historical_median,
        np.nan,
    )
    return out

def selected_month_summary(calendar:pd.DataFrame,year:int,month:int)->dict:
    frame=monthly_anomalies(calendar)
    picked=frame.loc[frame.year.eq(year)&frame.month.eq(month)]
    if len(picked)!=1:raise ValueError("Selected month/year not present")
    item=picked.iloc[0]
    comparable=frame.loc[frame.month.eq(month)&frame.value.notna()]
    if np.isfinite(item.value):
        rank=int(comparable.value.rank(ascending=False,method="min").loc[
            comparable.year.eq(year)].iloc[0])
    else:rank=None
    return dict(value=float(item.value),median=float(item.historical_median),
                anomaly_pct=float(item.anomaly_pct),rank=rank,
                comparable_years=len(comparable),source=str(item.source))
