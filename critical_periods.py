"""FireLens V9.1: cautious exploratory same-season *recorded-hotspot* activity screening.

Inputs are positive-record cell counts and coarse day-level flags, NOT actual
observation coverage, burned area, fire occurrence, or verified negatives.

We only rank days with positive records from BOTH sensors and no documented
S-NPP anomaly-window flag. That filter improves interpretability but does NOT
establish that the satellites observed every cell, nor calibrate sensor sensitivity.

Seasonal references use OTHER years in a +/- 14-day circular calendar window;
references exclude days lacking positive records from both sensors. The
quantile is DESCRIPTIVE; it is not a statistical significance threshold.
"""

import numpy as np
import pandas as pd

REQUIRED = {
    "target_date_utc", "target_positive_cells",
    "target_modis_has_any_hotspot_record",
    "target_viirs_has_any_hotspot_record",
    "target_snpp_known_anomaly_window",
}


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    # Avoid bool("False") == True for CSV-derived text fields.
    return s.astype("string").str.lower().str.strip().isin({"true", "1", "yes"}).fillna(False)


def build_daily_evidence(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC target day; derived evidence-quality labels."""
    missing = REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    m = manifest.copy()
    m["date"] = pd.to_datetime(m["target_date_utc"], errors="raise").dt.normalize()
    if m["date"].isna().any() or m["date"].duplicated().any():
        raise ValueError("UTC target dates must be present and unique")
    m = m.sort_values("date").reset_index(drop=True)
    m["occupied_cells"] = pd.to_numeric(m["target_positive_cells"], errors="raise")
    if (m["occupied_cells"] < 0).any():
        raise ValueError("Occupied cell counts cannot be negative")

    modis = _as_bool(m["target_modis_has_any_hotspot_record"])
    viirs = _as_bool(m["target_viirs_has_any_hotspot_record"])
    anomaly = _as_bool(m["target_snpp_known_anomaly_window"])
    m["record_context"] = np.select(
        [anomaly, (~modis & ~viirs), (~modis | ~viirs)],
        ["Known S-NPP anomaly window", "Neither sensor has a positive record", "Only one sensor has a positive record"],
        default="Both sensors have positive records",
    )
    m["comparison_eligible"] = (modis & viirs & ~anomaly).astype(bool)
    m["year"] = m["date"].dt.year
    # A leap-year reference keeps Feb 29 as its own day, and preserves March-Dec
    # alignment between leap and non-leap calendar years.
    m["season_day"] = pd.to_datetime("2000-" + m["date"].dt.strftime("%m-%d")).dt.dayofyear
    return m


def seasonal_screen(manifest: pd.DataFrame, *, window_days: int = 14,
                    peer_quantile: float = 0.90,
                    ratio_threshold: float = 1.25,
                    min_occupied_cells: int = 20,
                    min_peers: int = 30,
                    min_streak: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily table and contiguous candidate high-activity periods.

    Peer periods use other calendar years only; no learned model or holdout
    performance claim is made. Comparing the other years can include future
    archival data relative to the selected historical date. This is retrospective
    pattern exploration, explicitly NOT a prospective forecast or leakage-free
    training feature.
    """
    if not 0 <= window_days <= 60:
        raise ValueError("window_days must be in [0,60]")
    if not (0 < peer_quantile < 1):
        raise ValueError("peer_quantile must be in (0,1)")
    if min_streak < 1 or min_peers < 1 or ratio_threshold < 1:
        raise ValueError("Check minimum streak, peer count and ratio")

    m = build_daily_evidence(manifest)
    counts = m["occupied_cells"].to_numpy(dtype=float)
    days = m["season_day"].to_numpy(dtype=int)
    years = m["year"].to_numpy(dtype=int)
    eligible = m["comparison_eligible"].to_numpy(dtype=bool)

    medians = np.full(len(m), np.nan)
    qvals = np.full(len(m), np.nan)
    n_peers = np.zeros(len(m), dtype=int)
    for i, season_day in enumerate(days):
        dist = np.abs(days - season_day)
        dist = np.minimum(dist, 366 - dist)
        peer_mask = (years != years[i]) & (dist <= window_days) & eligible
        peers = counts[peer_mask]
        n_peers[i] = len(peers)
        if len(peers) >= min_peers:
            medians[i] = float(np.median(peers))
            qvals[i] = float(np.quantile(peers, peer_quantile))

    m["peer_days"] = n_peers
    m["peer_median"] = medians
    m["peer_upper_quantile"] = qvals
    m["relative_to_peer_median"] = np.divide(counts, np.maximum(medians, 1.0), out=np.full(len(m),np.nan), where=np.isfinite(medians))
    m["excess_over_peer_median"] = np.where(np.isfinite(medians), np.maximum(counts-medians,0),np.nan)
    m["candidate_day"] = (
        eligible & np.isfinite(qvals) &
        (counts >= min_occupied_cells) &
        (counts > qvals) &
        (counts >= ratio_threshold * np.maximum(medians, 1))
    )

    # A critical-period candidate needs consecutive candidate days, not just a
    # one-off high count. Gaps with uncertain evidence do not bridge periods.
    candidates = m.loc[m["candidate_day"]].copy()
    fields = ["start_date", "end_date", "days", "peak_date", "peak_occupied_cells",
              "sum_excess_cell_days", "median_peer_reference", "peak_ratio", "status"]
    if len(candidates) == 0:
        # Keep datetime dtypes even if no candidate dates survive screening.
        empty=pd.DataFrame({field:pd.Series(dtype='object') for field in fields})
        for field in ('start_date','end_date','peak_date'):
            empty[field]=pd.Series(dtype='datetime64[ns]')
        return m, empty
    new_period = candidates["date"].diff().dt.days.ne(1)
    candidates["period_id"] = new_period.cumsum()
    events = []
    for _, block in candidates.groupby("period_id"):
        if len(block) < min_streak:
            continue
        peak = block.loc[block["occupied_cells"].idxmax()]
        events.append({
            "start_date": block["date"].iloc[0],
            "end_date": block["date"].iloc[-1],
            "days": len(block),
            "peak_date": peak["date"],
            "peak_occupied_cells": int(peak["occupied_cells"]),
            "sum_excess_cell_days": float(block["excess_over_peer_median"].sum()),
            "median_peer_reference": float(block["peer_median"].median()),
            "peak_ratio": float(block["relative_to_peer_median"].max()),
            "status": "Candidate unusually high recorded activity (retrospective)",
        })
    periods = pd.DataFrame(events,columns=fields)
    for field in ('start_date','end_date','peak_date'):
        periods[field]=pd.to_datetime(periods[field],errors='coerce')
    if len(periods):
        periods = periods.sort_values(["sum_excess_cell_days","start_date"],ascending=[False,True]).reset_index(drop=True)
    return m, periods