"""Leave-one-complete-overlap-year-out validation of satellite archive count mappings.

Median absolute % error and median signed % bias require actual NOAA20 >0;
Spearman includes finite predictions with observed zero. MODIS-zero inputs
remain NaN, not pseudo-negative observations.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from v15.calibration import (
    ROOT,SOURCE,REGIONS,METHODS,country_month_rows,
    complete_overlap_years,fit_calibration,available_pairs,
)

VALIDATION_CSV=ROOT/"v15/calibration_validation.csv"

def score_frame(frame:pd.DataFrame,method:str,continent:str)->dict:
    valid=frame.loc[frame.predicted.notna()&frame.viirs.notna()].copy()
    percent=valid.loc[valid.viirs.gt(0)]
    ape=((percent.predicted-percent.viirs).abs()/percent.viirs*100).to_numpy(float)
    bias=((percent.predicted-percent.viirs)/percent.viirs*100).to_numpy(float)
    corr=float(spearmanr(valid.predicted,valid.viirs).statistic) if (
        len(valid)>1 and valid.predicted.nunique()>1 and valid.viirs.nunique()>1
    ) else np.nan
    return dict(
        level="global" if continent=="GLOBAL" else "continent",
        continent=continent,method=method,n_source_pairs=len(frame),
        n_predictions=len(valid),n_pct_valid=len(percent),
        n_modis_zero=int(frame.modis.eq(0).sum()),
        median_abs_pct_error=float(np.median(ape)) if len(ape) else np.nan,
        median_bias_pct=float(np.median(bias)) if len(bias) else np.nan,
        spearman=corr,
    )

def validation_predictions(rows:pd.DataFrame)->pd.DataFrame:
    years=complete_overlap_years(rows)
    if len(years)<3:
        raise ValueError("At least three complete-overlap years required")
    frames=[]
    for holdout in years:
        train=tuple(y for y in years if y!=holdout)
        model=fit_calibration(rows,years=train)
        test=available_pairs(rows,years=(holdout,))
        for method in METHODS:
            predicted=model.predict(test,method)
            frames.append(predicted[["iso3","country","continent","year","month",
                                     "modis","viirs","predicted","method"]])
    return pd.concat(frames,ignore_index=True)

def summarize_predictions(predictions:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for method in METHODS:
        subset=predictions.loc[predictions.method.eq(method)]
        rows.append(score_frame(subset,method,"GLOBAL"))
        for continent,region in subset.groupby("continent",sort=True):
            rows.append(score_frame(region,method,str(continent)))
    summary=pd.DataFrame(rows)
    summary["train_strategy"]="leave-one-calendar-year-out; complete overlap years only"
    summary["source_sensor"]="MODIS Terra/Aqua"
    summary["target_sensor"]="VIIRS NOAA-20"
    summary["holdout_years"]=",".join(str(y) for y in sorted(predictions.year.unique()))
    return summary

def selected_method(summary:pd.DataFrame)->str:
    global_scores=summary.loc[summary.level.eq("global")]
    eligible=global_scores.loc[global_scores.n_pct_valid.gt(0)&
                              global_scores.median_abs_pct_error.notna()].copy()
    if eligible.empty:
        raise ValueError("No held-out percentage errors")
    eligible["tie_rank"]=eligible.method.map({x:i for i,x in enumerate(METHODS)})
    return str(eligible.sort_values(["median_abs_pct_error","tie_rank"]).iloc[0].method)

def prediction_intervals(predictions:pd.DataFrame,method:str)->dict[str,tuple[float,float]]:
    chosen=predictions.loc[
        predictions.method.eq(method)&predictions.predicted.gt(0)&
        predictions.viirs.notna()
    ].copy()
    chosen["log_residual"]=np.log1p(chosen.viirs)-np.log1p(chosen.predicted)
    if len(chosen)<100:
        raise ValueError("Insufficient held-out observations for interval")
    def q(v):
        return tuple(float(x) for x in np.quantile(v,[.1,.9]))
    bands={"GLOBAL":q(chosen.log_residual)}
    for continent,region in chosen.groupby("continent"):
        if len(region)>=100:
            bands[str(continent)]=q(region.log_residual)
    return bands

def run(source:Path=SOURCE,region_path:Path=REGIONS):
    rows=country_month_rows(source,region_path)
    predictions=validation_predictions(rows)
    summary=summarize_predictions(predictions)
    best=selected_method(summary)
    return summary,predictions,best

if __name__=="__main__":
    summary,points,winner=run()
    VALIDATION_CSV.parent.mkdir(parents=True,exist_ok=True)
    summary.to_csv(VALIDATION_CSV,index=False,float_format="%.8f")
    print("held-out years:",sorted(points.year.unique()))
    print("selected method:",winner)
    print(summary.loc[summary.level.eq("global")].to_string(index=False))
