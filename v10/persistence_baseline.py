#!/usr/bin/env python3
"""Historical next-day recorded-hotspot persistence benchmark.

Prediction for target day T: previous UTC day's union positive 10-km cells.
Only positive-cell overlap is scored. No satellite coverage/verified negatives,
forecasted fire perimeter, or independent truth is asserted.

Usage:
 python v10/persistence_baseline.py \
   --daily v10/processed/amazon/daily_summary.csv \
   --grid /path/to/grid_10km_daily.csv.gz \
   --out /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import gzip
from itertools import groupby
from pathlib import Path

import pandas as pd

S_NPP_ANOMALY = [
    ("2024-07-09", "2024-07-16"),
    ("2024-07-24", "2024-07-29"),
]


def is_anomaly(day: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) <= day <= pd.Timestamp(b) for a,b in S_NPP_ANOMALY)


def compute(daily_path: Path, grid_path: Path, output: Path):
    """Stream sorted grid rows day by day without loading all cells into RAM."""
    daily = pd.read_csv(daily_path)
    if daily.empty:
        raise ValueError("Empty daily manifest")
    daily["date"] = pd.to_datetime(daily["date_utc"], errors="raise")
    daily = daily.sort_values("date").reset_index(drop=True)
    if daily.date.duplicated().any():
        raise ValueError("Duplicate target dates")
    if not daily.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Input manifest must cover consecutive UTC dates")

    output.mkdir(parents=True, exist_ok=True)
    rows=[]
    prior=set()
    prior_recorded=False
    with gzip.open(grid_path, "rt", encoding="utf-8", newline="") as fh:
        groups=groupby(csv.DictReader(fh),key=lambda item: item["date_utc"])
        nxt=next(groups,None)
        for n,r in daily.iterrows():
            day=r["date_utc"]
            if nxt is not None and nxt[0] < day:
                raise ValueError(f"Grid date {nxt[0]} absent from manifest")
            today=set()
            if nxt is not None and nxt[0] == day:
                today={(int(q["grid_x"]),int(q["grid_y"])) for q in nxt[1]}
                nxt=next(groups,None)
            if len(today) != int(r["union_positive_cells"]):
                raise ValueError(f"Grid/daily source mismatch for {day}")
            tp=len(today & prior)
            fp=len(prior - today)
            fn=len(today - prior)
            if tp+fn != len(today):
                raise ValueError("Actual-cell consistency failed")
            date=r["date"]
            both_today = bool(r["modis_has_positive_records"]) and bool(r["viirs_has_positive_records"])
            eligible = (n > 0 and both_today and prior_recorded
                        and not is_anomaly(date)
                        and not is_anomaly(date-pd.Timedelta(days=1)))
            denom=2*tp+fp+fn
            rows.append({
                "target_date_utc":day,"year":int(r["year"]),
                "tp_positive_cells":tp,"fp_positive_cells":fp,"fn_positive_cells":fn,
                "f1_recorded_positive_overlap":round(2*tp/denom,6) if denom else 0.,
                "precision_proxy":round(tp/(tp+fp),6) if tp+fp else 0.,
                "recall_proxy":round(tp/(tp+fn),6) if tp+fn else 0.,
                "both_sensor_record_eligible":eligible,
            })
            prior=today
            prior_recorded=both_today
        if nxt is not None:
            raise ValueError(f"Grid date {nxt[0]} after manifest end")

    table=pd.DataFrame(rows)
    table.to_csv(output / "persistence_baseline_daily.csv",index=False)
    summary=[]
    for year,year_rows in table.groupby("year"):
        for group, subset in [
            ("all_days",year_rows.iloc[1:] if year == table.year.iloc[0] else year_rows),
            ("both_record_eligible",year_rows.loc[year_rows.both_sensor_record_eligible]),
        ]:
            tp=int(subset.tp_positive_cells.sum())
            fp=int(subset.fp_positive_cells.sum())
            fn=int(subset.fn_positive_cells.sum())
            summary.append({
                "year":int(year),"subset":group,"days":len(subset),
                "tp":tp,"fp":fp,"fn":fn,
                "f1_recorded_positive_overlap":round(2*tp/(2*tp+fp+fn),6) if 2*tp+fp+fn else 0.,
                "iou_recorded_positive_overlap":round(tp/(tp+fp+fn),6) if tp+fp+fn else 0.,
            })
    annual=pd.DataFrame(summary)
    annual.to_csv(output / "persistence_baseline_annual.csv",index=False)
    return table,annual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily",type=Path,required=True)
    parser.add_argument("--grid",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    _, annual=compute(args.daily,args.grid,args.out)
    print(annual.to_string(index=False))


if __name__=="__main__":
    main()
