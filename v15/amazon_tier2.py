"""Amazon Tier-2: three methods comparing daily records versus positive 10-km cells.

Train on fully dated 2022–2024, hold out fully dated 2025. Exclude the
documented July 2024 S-NPP anomaly days from fitting. VIIRS here is
Suomi NPP, not the global NOAA-20 archive. Only positive observations
are in these files; no sensor observation-opportunity calibration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"v10/processed/amazon/daily_summary.csv"
STATUS=ROOT/"data/historic/july_2024_sensor_status.csv"
METHODS=("none","ratio","log")
UNITS={
    "records":("modis_record_rows","viirs_record_rows"),
    "positive_cells":("modis_positive_cells","viirs_positive_cells")
}

def complete_years(frame):
    necessary={"date_utc","year","month"}
    for cols in UNITS.values():necessary.update(cols)
    if missing:=necessary-set(frame.columns):
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if frame.date_utc.duplicated().any():raise ValueError("Duplicate UTC dates")
    stamps=pd.to_datetime(frame.date_utc,errors="raise")
    years=[]
    for year,group in frame.groupby("year"):
        expected=pd.date_range(f"{int(year)}-01-01",f"{int(year)}-12-31")
        present=pd.DatetimeIndex(stamps[group.index])
        if len(group)==len(expected) and present.nunique()==len(expected) and present.isin(expected).all():
            years.append(int(year))
    return tuple(sorted(years))

def outage_dates(status):
    required={"UTC date","S-NPP anomaly window","FIRMS listed VIIRS missing date"}
    if missing:=required-set(status.columns):
        raise ValueError(f"Missing outage metadata: {sorted(missing)}")
    mask=(status["S-NPP anomaly window"].astype(str).str.lower().eq("true") |
          status["FIRMS listed VIIRS missing date"].astype(str).str.lower().eq("true"))
    return set(status.loc[mask,"UTC date"].astype(str))

def _design(m,month):
    return np.column_stack([np.ones(len(m)),np.log1p(m)]+
                           [(month==k).astype(float) for k in range(2,13)])

def fit(train,unit,flags):
    mcol,vcol=UNITS[unit]
    clean=train.loc[~train.date_utc.isin(flags)&train[mcol].gt(0)].copy()
    if len(clean)<100:raise ValueError("Too few clean training days")
    ratios={int(m):float(g[vcol].sum()/g[mcol].sum())
            for m,g in clean.groupby("month")}
    x=_design(clean[mcol].to_numpy(float),clean.month.to_numpy(int))
    y=np.log1p(clean[vcol].to_numpy(float))
    penalty=np.eye(x.shape[1])*3.
    penalty[0,0]=.01;penalty[1,1]=.05
    beta=np.linalg.solve(x.T@x+penalty,x.T@y)
    return dict(ratios=ratios,beta=beta,train_days=len(clean))

def predict(test,unit,model,method):
    if method not in METHODS:raise ValueError("Unknown method")
    m=test[UNITS[unit][0]].to_numpy(float)
    month=test.month.to_numpy(int)
    active=m>0
    out=np.full(len(test),np.nan)
    if method=="none":out[active]=m[active]
    elif method=="ratio":
        ratio=np.array([model["ratios"].get(int(k),np.nan) for k in month])
        out[active]=m[active]*ratio[active]
    else:
        x=_design(m[active],month[active])
        out[active]=np.maximum(0,np.expm1(np.clip(x@model["beta"],-20,30)))
    return out

def evaluate(actual,predicted):
    real=np.asarray(actual,dtype=float)
    pred=np.asarray(predicted,dtype=float)
    finite=np.isfinite(real)&np.isfinite(pred)
    valid=finite&(real>0)
    ape=100*np.abs(pred[valid]-real[valid])/real[valid]
    bias=100*(pred[valid]-real[valid])/real[valid]
    correlation=float(spearmanr(pred[finite],real[finite]).statistic) if finite.sum()>1 else np.nan
    return dict(n_days=len(real),n_evaluated=int(valid.sum()),
                n_undefined=int((~finite).sum()),
                median_abs_pct_error=float(np.median(ape)) if len(ape) else np.nan,
                median_bias_pct=float(np.median(bias)) if len(bias) else np.nan,
                spearman=correlation)

def validate(data=None,status=None):
    if data is None:data=pd.read_csv(DATA)
    if status is None:status=pd.read_csv(STATUS)
    years=complete_years(data)
    if years!=(2022,2023,2024,2025):
        raise ValueError(f"Expected full source calendars in 2022–25, got {years}")
    flags=outage_dates(status)
    train=data.loc[data.year.lt(2025)]
    holdout=data.loc[data.year.eq(2025)]
    result=dict(source_years=list(years),train_years=[2022,2023,2024],
                holdout_year=2025,
                known_s_npp_flagged_dates_in_training=len(set(train.date_utc)&flags),
                units={})
    for unit,(_,vcol) in UNITS.items():
        model=fit(train,unit,flags)
        metrics={method:evaluate(holdout[vcol],predict(holdout,unit,model,method))
                 for method in METHODS}
        winner=min(METHODS,key=lambda m:metrics[m]["median_abs_pct_error"])
        result["units"][unit]=dict(methods=metrics,winning_method=winner,
                                   fitted_days=int(model["train_days"]))
    return result
