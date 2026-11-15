"""Seeded 2025 Amazon placebo: hide VIIRS positives for 30 five-day windows.

Uses Tier-2 winner trained on 2022–24 and 80% nominal intervals from
out-of-year residuals within 2022–24. The selection criterion is positive
VIIRS reports and no known disruption flag; it does NOT certify complete
satellite observation opportunity or a real no-outage window.
"""
import numpy as np
import pandas as pd
from v15.amazon_tier2 import (
    DATA,STATUS,METHODS,UNITS,outage_dates,fit,predict,evaluate
)

SEED=20261115

def placebo(data=None,status=None,tier2=None):
    if data is None:data=pd.read_csv(DATA)
    if status is None:status=pd.read_csv(STATUS)
    if tier2 is None:
        from v15.amazon_tier2 import validate
        tier2=validate(data,status)
    flagged=outage_dates(status)
    train=data.loc[data.year.lt(2025)]
    held=data.loc[data.year.eq(2025)].sort_values("date_utc").reset_index(drop=True)
    valid=(~held.date_utc.isin(flagged)&held.viirs_record_rows.gt(0)&
           held.modis_record_rows.gt(0))
    starts=[i for i in range(len(held)-4) if valid.iloc[i:i+5].all()]
    random=np.random.default_rng(SEED)
    random.shuffle(starts)
    chosen=[]
    used=set()
    for start in starts:
        positions=set(range(start,start+5))
        if used&positions:continue
        chosen.append(start)
        used.update(positions)
        if len(chosen)==30:break
    if len(chosen)<30:
        raise ValueError("Could not find 30 nonoverlapping positive-report windows")
    test=held.iloc[sorted(used)]
    result=dict(seed=SEED,window_count=30,window_length_days=5,
                sampled_day_count=len(test),
                window_starts=sorted(held.date_utc.iloc[chosen].tolist()),
                selection_note="Positive MODIS and VIIRS S-NPP records on each sampled day; no known flag in the available outage table. Full observation opportunity unverified.")
    for unit,(_,vcol) in UNITS.items():
        method=tier2["units"][unit]["winning_method"]
        model=fit(train,unit,flagged)
        pred=predict(test,unit,model,method)
        actual=test[vcol].to_numpy(float)
        residuals=[]
        for year in (2022,2023,2024):
            fitting=fit(train.loc[train.year.ne(year)],unit,flagged)
            withheld=train.loc[train.year.eq(year)&~train.date_utc.isin(flagged)]
            estimated=predict(withheld,unit,fitting,method)
            observed=withheld[vcol].to_numpy(float)
            valid=np.isfinite(estimated)&(estimated>=0)&np.isfinite(observed)
            residuals.extend(
                (np.log1p(observed[valid])-np.log1p(estimated[valid])).tolist()
            )
        low,high=np.quantile(residuals,[.1,.9])
        lower=np.maximum(0,np.expm1(np.log1p(pred)+low))
        upper=np.expm1(np.log1p(pred)+high)
        eligible=np.isfinite(pred)&(actual>0)
        metrics=evaluate(actual,pred)
        metrics.update(
            coverage_80pct_nominal=float(np.mean(
                (actual[eligible]>=lower[eligible])&
                (actual[eligible]<=upper[eligible])
            )),
            residual_quantile_low=float(low),
            residual_quantile_high=float(high),
            winning_method=method,
            no_fill_applied=True,
        )
        result[unit]=metrics
    return result
