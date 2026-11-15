"""Tier-1 statistical MODIS → VIIRS NOAA-20 monthly archived-record calibration.

Counts are NASA FIRMS type-0 POSITIVE RECORDS, not active grid-cell days,
satellite observation opportunities, or calibrated physical fire incidence.
NOAA-20 begins Apr 2018; months with globally absent NOAA-20 archive
records are excluded. Months with positives still do not prove full coverage.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"v11/processed/global/global_country_year.csv"
REGIONS=ROOT/"v15/iso_regions.csv"
MONTH_COLS=[f"m{i:02d}" for i in range(1,13)]
METHODS=("none","ratio","log","global_ratio")
MIN_COUNTRY_MONTHS=36
COUNTRY_PRIOR_STRENGTH=24.0
REGION_PRIOR_STRENGTH=48.0

def iso_continents(path:Path=REGIONS)->dict[str,str]:
    source=pd.read_csv(path,dtype=str).fillna("")
    required={"alpha-3","region","sub-region","intermediate-region"}
    if missing:=required.difference(source.columns):
        raise ValueError(f"ISO metadata missing: {sorted(missing)}")
    mapping={}
    for _,r in source.iterrows():
        iso=r["alpha-3"].strip().upper()
        region=r["region"].strip()
        intermediate=r["intermediate-region"].strip()
        if region=="Americas":
            continent="South America" if intermediate=="South America" else "North America"
        elif region in {"Africa","Asia","Europe","Oceania","Antarctica"}:
            continent=region
        else:
            continent="Unmapped"
        if iso:
            mapping[iso]=continent
    return mapping

def country_month_rows(path:Path=SOURCE,region_path:Path=REGIONS)->pd.DataFrame:
    raw=pd.read_csv(path)
    required={"sensor","year","country","iso3","vegetation_records",*MONTH_COLS}
    if missing:=required.difference(raw.columns):
        raise ValueError(f"Country-year archive missing columns: {sorted(missing)}")
    raw=raw.loc[raw.sensor.isin(("MODIS","VIIRS_NOAA20"))].copy()
    if raw.duplicated(["sensor","year","iso3"]).any():
        raise ValueError("Duplicate source country/year/sensor rows")
    if not raw[MONTH_COLS].apply(pd.to_numeric,errors="raise").ge(0).all().all():
        raise ValueError("Negative monthly record count")
    if not raw[MONTH_COLS].sum(axis=1).eq(raw.vegetation_records).all():
        raise ValueError("Monthly sum differs from type-0 source total")
    raw["continent"]=raw.iso3.map(iso_continents(region_path)).fillna("Unmapped")
    melted=raw.melt(id_vars=["sensor","year","iso3","country","continent"],
                    value_vars=MONTH_COLS,var_name="month_code",value_name="records")
    melted["month"]=melted.month_code.str[1:].astype(int)
    wide=melted.pivot_table(index=["iso3","year","month"],columns="sensor",
                            values="records",aggfunc="sum",dropna=True).reset_index()
    geo=raw.sort_values("year").drop_duplicates("iso3").set_index("iso3")[["country","continent"]]
    wide["country"]=wide.iso3.map(geo.country)
    wide["continent"]=wide.iso3.map(geo.continent).fillna("Unmapped")
    wide=wide.rename(columns={"MODIS":"modis","VIIRS_NOAA20":"viirs"})
    return wide[["iso3","country","continent","year","month","modis","viirs"]].sort_values(
        ["iso3","year","month"]).reset_index(drop=True)

def complete_overlap_years(rows:pd.DataFrame)->tuple[int,...]:
    paired=rows.loc[rows.modis.notna()&rows.viirs.notna()]
    totals=paired.groupby(["year","month"])[["modis","viirs"]].sum()
    eligible=[]
    for year,group in totals.groupby(level="year"):
        months=set(group.index.get_level_values("month"))
        if months==set(range(1,13)) and group.modis.gt(0).all() and group.viirs.gt(0).all():
            eligible.append(int(year))
    return tuple(sorted(eligible))

def available_pairs(rows:pd.DataFrame,*,years:tuple[int,...]|None=None)->pd.DataFrame:
    years=complete_overlap_years(rows) if years is None else years
    return rows.loc[rows.year.isin(years)&rows.modis.notna()&rows.viirs.notna()].copy()

def design(frame:pd.DataFrame)->np.ndarray:
    month=frame.month.to_numpy(dtype=int)
    n=len(frame)
    return np.column_stack([np.ones(n),
        np.log1p(frame.modis.to_numpy(dtype=float))]+
        [(month==m).astype(float) for m in range(2,13)])

def ridge_fit(x:np.ndarray,y:np.ndarray,*,prior:np.ndarray|None=None,
              shrink:float=3.0)->np.ndarray:
    size=x.shape[1]
    default_prior=prior is None
    if prior is None:prior=np.zeros(size)
    reg=np.eye(size)*float(shrink)
    reg[0,0]=.01
    if default_prior:reg[1,1]=.05
    return np.linalg.solve(x.T@x+reg,x.T@y+reg@prior)

@dataclass
class FittedCalibration:
    eligible_years:tuple[int,...]
    ratios_country:dict
    ratios_region:dict
    ratios_global:dict
    single_global_ratio:float
    beta_global:np.ndarray
    beta_region:dict
    beta_country:dict
    zero_probability:dict
    zero_probability_global:float

    def predict(self,frame:pd.DataFrame,method:str)->pd.DataFrame:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        if missing:={"iso3","continent","month","modis"}.difference(frame.columns):
            raise ValueError(f"Prediction missing columns: {sorted(missing)}")
        out=frame.copy()
        values=np.full(len(out),np.nan)
        probs=np.full(len(out),np.nan)
        for i,(_,r) in enumerate(out.iterrows()):
            m=float(r.modis)
            if not np.isfinite(m):continue
            if m==0:
                probs[i]=self.zero_probability.get(
                    (str(r.continent),int(r.month)),self.zero_probability_global)
                continue
            if m<0:raise ValueError("Negative MODIS count")
            code=str(r.iso3);region=str(r.continent);month=int(r.month)
            if method=="none":
                pred=m
            elif method=="global_ratio":
                pred=m*self.single_global_ratio
            elif method=="ratio":
                ratio=self.ratios_country.get((code,month))
                if ratio is None:
                    ratio=self.ratios_region.get((region,month),self.ratios_global.get(month))
                pred=m*ratio if ratio is not None else np.nan
            else:
                beta=self.beta_country.get(code,self.beta_region.get(region,self.beta_global))
                feature=np.zeros(13)
                feature[:2]=[1,np.log1p(m)]
                if month>=2:feature[month]=1
                pred=np.expm1(np.clip(feature@beta,-20,30))
            values[i]=max(0,float(pred)) if np.isfinite(pred) else np.nan
        out["predicted"]=values
        out["p_viirs_positive_when_modis_zero"]=probs
        out["method"]=method
        return out

def fit_calibration(rows:pd.DataFrame,*,years:tuple[int,...]|None=None)->FittedCalibration:
    eligible=complete_overlap_years(rows) if years is None else tuple(sorted(set(years)))
    pairs=available_pairs(rows,years=eligible)
    if pairs.empty:raise ValueError("No eligible paired overlap-year records")
    positive=pairs.loc[pairs.modis.gt(0)].copy()
    if len(positive)<24:raise ValueError("At least 24 positive MODIS pairs required")
    def ratios(keys):
        totals=positive.groupby(keys)[["modis","viirs"]].sum()
        values=totals.viirs/totals.modis
        return {tuple(k) if isinstance(k,tuple) else k:float(v)
                for k,v in values.items() if np.isfinite(v)}
    country=ratios(["iso3","month"])
    region=ratios(["continent","month"])
    global_ratios=ratios(["month"])
    single_global_ratio=float(positive.viirs.sum()/positive.modis.sum())
    beta_global=ridge_fit(design(positive),np.log1p(positive.viirs.to_numpy(dtype=float)))
    beta_region={}
    for continent,g in positive.groupby("continent"):
        if len(g)<24:continue
        beta_region[str(continent)]=ridge_fit(
            design(g),np.log1p(g.viirs.to_numpy(dtype=float)),
            prior=beta_global,shrink=REGION_PRIOR_STRENGTH)
    beta_country={}
    for code,g in positive.groupby("iso3"):
        if len(g)<MIN_COUNTRY_MONTHS:continue
        prior=beta_region.get(str(g.continent.iloc[0]),beta_global)
        beta_country[str(code)]=ridge_fit(
            design(g),np.log1p(g.viirs.to_numpy(dtype=float)),
            prior=prior,shrink=COUNTRY_PRIOR_STRENGTH)
    zero=pairs.loc[pairs.modis.eq(0)]
    zero_by_month={}
    for (continent,month),g in zero.groupby(["continent","month"]):
        zero_by_month[(str(continent),int(month))]=float((g.viirs.gt(0).sum()+1)/(len(g)+2))
    global_prob=float((zero.viirs.gt(0).sum()+1)/(len(zero)+2))
    return FittedCalibration(eligible,country,region,global_ratios,single_global_ratio,beta_global,
                             beta_region,beta_country,zero_by_month,global_prob)
