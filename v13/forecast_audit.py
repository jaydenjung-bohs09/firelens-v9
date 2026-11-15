"""Reproduce FireLens exploratory 2025 next-day recorded-hotspot comparisons.

Usage:
    python v13/forecast_audit.py
    python v13/forecast_audit.py --json

Audits the frozen ConvLSTM archive. Same 285 target dates for all methods:
first seven target days excluded so 7-day historical baselines use only
past *recorded positive* grids. No test-dependent hyperparameter selection;
the >=3-of-7 fixed baseline is a predeclared simple alternative.
Bootstrap draws circular, consecutive 14-day blocks of target dates.
This is exploratory model-vs-archive label overlap, NOT fire forecast accuracy.
The period was inspected during development and is not an untouched holdout.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NPZ=ROOT/"data"/"predict"/"test_predictions.npz"
LOCKED_THRESHOLD=0.675
EXCLUDE_FIRST=7
BOOTSTRAP_REPLICATES=3000
BOOTSTRAP_BLOCK_DAYS=14
BOOTSTRAP_SEED=20261115


def _f1(totals):
    tp,fp,fn=np.asarray(totals,dtype=float)
    den=2*tp+fp+fn
    return float(2*tp/den) if den else 0.

def compute(path: Path=NPZ, bootstrap_replicates: int=BOOTSTRAP_REPLICATES):
    with np.load(path,allow_pickle=False) as data:
        scores=np.asarray(data["predicted_detection_proxy_scores"])
        actual=np.asarray(data["actual_recorded_hotspot_labels"],dtype=bool)
        dates=np.asarray(data["target_dates"],dtype=str)
    if scores.shape!=actual.shape or len(actual)!=292 or actual.ndim!=3:
        raise ValueError("Frozen test array shapes differ from original audit")
    # Verify the full historical reporting number before filtering common dates.
    all_pred=scores>=LOCKED_THRESHOLD
    all_tp=int(np.count_nonzero(all_pred & actual))
    all_fp=int(np.count_nonzero(all_pred & ~actual))
    all_fn=int(np.count_nonzero(~all_pred & actual))
    if (all_tp,all_fp,all_fn)!=(3581,4363,5480):
        raise ValueError("Frozen full-period ConvLSTM reference has changed")

    n=len(actual)-EXCLUDE_FIRST
    obs=actual[EXCLUDE_FIRST:]
    past=np.stack([
        actual[EXCLUDE_FIRST-k:len(actual)-k]
        for k in range(1,EXCLUDE_FIRST+1)
    ])
    forecasts={
        "ConvLSTM":all_pred[EXCLUDE_FIRST:],
        "One-day persistence":actual[EXCLUDE_FIRST-1:-1],
        "Seven-day union":past.any(axis=0),
        "At least three of seven days":past.sum(axis=0)>=3,
    }
    # Define day-level confusion summaries to preserve temporal grouping.
    by_day={}
    aggregate={}
    for label,pred in forecasts.items():
        by_day[label]=np.asarray(
            [[np.count_nonzero(p & t),np.count_nonzero(p & ~t),
              np.count_nonzero(~p & t)] for p,t in zip(pred,obs)],
            dtype=np.int64)
        tp,fp,fn=map(int,by_day[label].sum(axis=0))
        aggregate[label]={"tp":tp,"fp":fp,"fn":fn,
                          "f1":_f1([tp,fp,fn])}

    rng=np.random.default_rng(BOOTSTRAP_SEED)
    nblocks=(n+BOOTSTRAP_BLOCK_DAYS-1)//BOOTSTRAP_BLOCK_DAYS
    starts=rng.integers(0,n,size=(bootstrap_replicates,nblocks))
    indices=((starts[:,:,None]+np.arange(BOOTSTRAP_BLOCK_DAYS))%n).reshape(
        bootstrap_replicates,-1)[:,:n]
    boot={}
    for label,day in by_day.items():
        stats=day[indices].sum(axis=1)
        denom=2*stats[:,0]+stats[:,1]+stats[:,2]
        vals=np.divide(2*stats[:,0],denom,
                       out=np.zeros(len(stats)),where=denom>0)
        boot[label]=vals
        aggregate[label]["ci95_f1"]=list(map(float,np.quantile(vals,[.025,.975])))

    for label in list(forecasts):
        if label=="ConvLSTM":
            continue
        d=boot["ConvLSTM"]-boot[label]
        aggregate[label]["model_minus_baseline_f1"]=(
            aggregate["ConvLSTM"]["f1"]-aggregate[label]["f1"])
        aggregate[label]["ci95_model_minus_baseline"]=list(
            map(float,np.quantile(d,[.025,.975])))
    return {
        "evaluation":"exploratory, previously inspected chronological 2025 test",
        "target":"recorded next-day MODIS-or-VIIRS S-NPP positive 10km cell",
        "common_dates":n,"first_date":str(dates[EXCLUDE_FIRST]),
        "last_date":str(dates[-1]),
        "locked_model_threshold":LOCKED_THRESHOLD,
        "bootstrap":{"method":"circular moving blocks of 14 consecutive days",
                     "replicates":bootstrap_replicates,
                     "block_days":BOOTSTRAP_BLOCK_DAYS,
                     "seed":BOOTSTRAP_SEED},
        "methods":aggregate,
        "interpretation":[
            "F1 scores overlap with archived positive-cell labels; these are not independently verified fire/no-fire cells.",
            "No satellite observation-opportunity mask is available.",
            "The previously inspected test period is exploratory, not an untouched holdout.",
            "First seven test dates were excluded consistently from every method."
        ]
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--json",action="store_true")
    args=parser.parse_args()
    result=compute()
    if args.json:
        print(json.dumps(result,indent=2))
    else:
        print(f"Common target dates: {result['common_dates']}")
        for k,m in result["methods"].items():
            print(f"{k:29} F1 {m['f1']:.4f} "
                  f"95% block-bootstrap CI {m['ci95_f1'][0]:.4f}"
                  f" to {m['ci95_f1'][1]:.4f}")

if __name__=="__main__":
    main()
