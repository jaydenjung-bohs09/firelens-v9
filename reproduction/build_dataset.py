#!/usr/bin/env python3
"""FireLens Predict: historical MODIS + VIIRS S-NPP positive-hotspot sequence builder.

Input NASA FIRMS CSV(s) or ZIP(s) with arbitrary archive suffixes, organized in any
subdirectories. This builder is experimental; negative pixels are UNOBSERVED / NO
RECORDED HOTSPOT, not validated non-fire cells or sensor true negatives.
"""
import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

DEFAULT_START = '2022-01-01'
DEFAULT_END = '2025-12-31'
CALIFORNIA_BBOX = [-124.5, 32.5, -114.0, 42.1]  # lon_min,lat_min,lon_max,lat_max
# Source: 2024 S-NPP interruptions documented in original FireLens V8 research.
KNOWN_SNPP_ANOMALY_DAYS = set(pd.date_range('2024-07-09', '2024-07-16').strftime('%Y-%m-%d')) | set(pd.date_range('2024-07-24', '2024-07-29').strftime('%Y-%m-%d'))
REQUIRED = {'latitude', 'longitude', 'acq_date'}


def classify_sensor(df, filename):
    """Only MODIS Terra/Aqua and VIIRS Suomi NPP are accepted."""
    sf = str(filename).lower()
    satellites = set(df['satellite'].dropna().astype(str).str.strip().str.lower().unique()) if 'satellite' in df else set()
    instruments = set(df['instrument'].dropna().astype(str).str.strip().str.lower().unique()) if 'instrument' in df else set()
    if satellites:
        if satellites <= {'terra', 'aqua'}:
            return 'MODIS'
        if satellites <= {'snpp', 'npp', 'suomi-npp', 'suomi npp', 'suomi_npp'}:
            return 'VIIRS'
        # Mixed or different spacecraft: never silently blend in NOAA-20 or NOAA-21.
        raise ValueError(f'Unexpected satellite values {sorted(satellites)[:10]} in {filename}; only Terra/Aqua MODIS and S-NPP VIIRS are supported.')
    if instruments == {'modis'}:
        return 'MODIS'
    if 'm-c61' in sf or 'modis' in sf:
        return 'MODIS'
    if 'sv-c2' in sf or 'snpp' in sf or 's-npp' in sf:
        return 'VIIRS'
    raise ValueError(f'Cannot safely recognize NASA satellite product in {filename}; ensure satellite column exists.')


def read_csv_bytes(data, filename):
    try:
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    except Exception:
        return None
    df.columns = df.columns.astype(str).str.strip().str.lower()
    if not REQUIRED.issubset(set(df.columns)):
        return None
    return df


def collect_files(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Input folder not found: {root}')
    csvs = []
    for file in sorted(root.rglob('*')):
        if not file.is_file():
            continue
        if file.suffix.lower() == '.csv':
            raw = file.read_bytes()
            df = read_csv_bytes(raw, str(file))
            if df is not None:
                csvs.append((str(file.relative_to(root)), df))
        elif file.suffix.lower() == '.zip':
            try:
                with zipfile.ZipFile(file) as z:
                    for name in sorted(z.namelist()):
                        if not name.lower().endswith('.csv'):
                            continue
                        try:
                            df = read_csv_bytes(z.read(name), f'{file.name}/{name}')
                        except Exception:
                            continue
                        if df is not None:
                            csvs.append((str(file.relative_to(root)) + '::' + name, df))
            except zipfile.BadZipFile:
                raise ValueError(f'Invalid ZIP: {file}')
    if not csvs:
        raise ValueError('No NASA FIRMS CSVs found. Upload MODIS Terra/Aqua and VIIRS S-NPP archive CSV or ZIP files.')
    return csvs


def build(input_root, outdir, start=DEFAULT_START, end=DEFAULT_END, km=10, history=7, bbox=None):
    if history < 1 or km <= 0:
        raise ValueError('history and km must be positive.')
    bbox = bbox or CALIFORNIA_BBOX
    lon0, lat0, lon1, lat1 = bbox
    if lon0 >= lon1 or lat0 >= lat1:
        raise ValueError('Invalid bounding box.')
    start_date, end_date = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = pd.date_range(start_date, end_date, freq='D')
    if len(days) <= history:
        raise ValueError('Date range must exceed history length.')

    inputs = collect_files(input_root)
    all_sensors = {'MODIS': [], 'VIIRS': []}
    logs = []
    for filename, df in inputs:
        sensor = classify_sensor(df, filename)
        original = len(df)
        sub = df.copy()
        sub['acq_date'] = pd.to_datetime(sub['acq_date'], errors='coerce').dt.normalize()
        for col in ['latitude', 'longitude']:
            sub[col] = pd.to_numeric(sub[col], errors='coerce')
        sub = sub.dropna(subset=['latitude', 'longitude', 'acq_date'])
        sub = sub[sub['acq_date'].between(start_date, end_date)]
        sub = sub[sub.latitude.between(lat0, lat1) & sub.longitude.between(lon0, lon1)]
        if 'acq_time' not in sub:
            sub['acq_time'] = ''
        if 'satellite' not in sub:
            sub['satellite'] = 'MODIS_unspecified' if sensor == 'MODIS' else 'SNPP'
        # FIRMS CSVs can overlap at file boundaries; only obvious duplicate observations removed.
        sub = sub[['latitude', 'longitude', 'acq_date', 'acq_time', 'satellite']].copy()
        all_sensors[sensor].append(sub)
        logs.append({'source': filename, 'sensor': sensor, 'raw_rows': original, 'rows_in_date_and_bbox': len(sub),
                     'min_date_in_scope': str(sub.acq_date.min().date()) if len(sub) else None,
                     'max_date_in_scope': str(sub.acq_date.max().date()) if len(sub) else None})
    if not all(all_sensors.values()):
        raise ValueError('Need at least one MODIS file and one VIIRS S-NPP file; check satellite values.')
    # Never silently fill an entirely absent year with apparent no-hotspot labels.
    if len(days) > 365:
        for sensor, tables in all_sensors.items():
            present_years = set(pd.concat(tables, ignore_index=True).acq_date.dt.year.dropna().astype(int))
            absent = sorted(set(range(start_date.year, end_date.year+1))-present_years)
            if absent:
                raise ValueError(f'{sensor} has no archive records for {absent}. Upload every requested year before generating training labels.')

    records = {}
    for sensor in ('MODIS', 'VIIRS'):
        data = pd.concat(all_sensors[sensor], ignore_index=True)
        data = data.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time', 'satellite'])
        records[sensor] = data
        if len(data) == 0:
            raise ValueError(f'{sensor} has zero in-scope records. Check requested years/area.')

    trans = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
    x0, y0 = trans.transform(lon0, lat0)
    x1, y1 = trans.transform(lon1, lat1)
    cell_m = km * 1000
    # Fixed geographic frame from user-declared bbox, not derived from post-split observations.
    xmin, xmax = math.floor(min(x0, x1)/cell_m), math.floor(max(x0, x1)/cell_m)
    ymin, ymax = math.floor(min(y0, y1)/cell_m), math.floor(max(y0, y1)/cell_m)
    h, w = ymax-ymin+1, xmax-xmin+1
    if h*w > 100_000:
        raise ValueError('Excessive raster size; increase grid resolution.')
    presence = np.zeros((len(days), h, w, 4), dtype=np.uint8)
    for channel, sensor in enumerate(('MODIS', 'VIIRS')):
        d = records[sensor]
        x, y = trans.transform(d.longitude.to_numpy(), d.latitude.to_numpy())
        cx = np.floor(np.asarray(x)/cell_m).astype(np.int32) - xmin
        cy = np.floor(np.asarray(y)/cell_m).astype(np.int32) - ymin
        t = (d.acq_date - start_date).dt.days.to_numpy(dtype=np.int32)
        valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h) & (t>=0) & (t < len(days))
        presence[t[valid], cy[valid], cx[valid], channel] = 1

    viirs_record_days = set(records['VIIRS'].acq_date)
    modis_record_days = set(records['MODIS'].acq_date)
    viirs_any = np.array([d in viirs_record_days for d in days], dtype=bool)
    modis_any = np.array([d in modis_record_days for d in days], dtype=bool)
    anomaly = np.array([d.strftime('%Y-%m-%d') in KNOWN_SNPP_ANOMALY_DAYS for d in days], dtype=bool)
    # Daily global indicators: no-record does not mean no observation opportunity.
    presence[:, :, :, 2] = viirs_any.astype(np.uint8)[:, None, None]
    presence[:, :, :, 3] = anomaly.astype(np.uint8)[:, None, None]
    label = np.maximum(presence[:, :, :, 0], presence[:, :, :, 1])
    # Next-day targets: history input [target-7, ..., target-1].
    t_idx = np.arange(history, len(days))
    X = np.stack([presence[t-history:t] for t in t_idx]).astype(np.uint8)
    y = label[t_idx].astype(np.uint8)
    dates = np.asarray(days[t_idx].strftime('%Y-%m-%d').to_numpy(), dtype='U10')
    n = len(dates)
    n_train_days = round(len(days) * 0.60)
    n_val_days = round(len(days) * 0.20)
    # Calendar-day 60/20/20 allocation from the previously agreed date boundaries.
    # Windows in the first seven days do not have a complete input history.
    split = np.where(t_idx < n_train_days, 'train',
                     np.where(t_idx < n_train_days + n_val_days, 'validation', 'test'))
    manifest = pd.DataFrame({
        'target_date_utc': dates,
        'history_start_utc': days[t_idx-history].strftime('%Y-%m-%d').to_numpy(),
        'history_end_utc': days[t_idx-1].strftime('%Y-%m-%d').to_numpy(),
        'split': split,
        'target_positive_cells': y.sum(axis=(1,2)).astype(int),
        'target_modis_has_any_hotspot_record': modis_any[t_idx],
        'target_viirs_has_any_hotspot_record': viirs_any[t_idx],
        'target_snpp_known_anomaly_window': anomaly[t_idx],
        'target_both_sensors_have_any_hotspot_record': modis_any[t_idx] & viirs_any[t_idx],
    })
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir/'firelens_sequences.npz', X=X, y=y, dates=dates)
    manifest.to_csv(outdir/'sequence_manifest.csv', index=False)
    (outdir/'source_inventory.csv').write_text(pd.DataFrame(logs).to_csv(index=False))
    counts = {s: int(len(records[s])) for s in records}
    card = {
        'project': 'FireLens Predict - experimental 7-day-to-next-day detected-hotspot forecast',
        'intended_date_range': [start, end], 'history_days': history,
        'grid_km': km, 'grid_projection': 'EPSG:6933', 'fixed_bbox_lonlat': bbox,
        'frame_note': 'Geographic bounding rectangle, not an exact California state polygon; input files must already be California-filtered.',
        'channels': ['MODIS_positive_hotspot', 'VIIRS_S_NPP_positive_hotspot', 'VIIRS_has_any_record_that_day_global_proxy', 'SNPP_known_2024_anomaly_window_indicator'],
        'target': 'Next-day MODIS-or-VIIRS S-NPP recorded positive hotspot in grid cell (not wildfire ignition).',
        'data_sources': logs, 'unique_records_by_sensor': counts,
        'shape_X': list(X.shape), 'shape_y': list(y.shape),
        'split_sizes': {s: int((split == s).sum()) for s in ['train','validation','test']},
        'calendar_day_split_boundaries': {
            'train': [start, str(days[n_train_days-1].date())],
            'validation': [str(days[n_train_days].date()), str(days[n_train_days+n_val_days-1].date())],
            'test': [str(days[n_train_days+n_val_days].date()), end],
        },
        'split_target_date_ranges': {s: [str(dates[split==s][0]), str(dates[split==s][-1])] if (split==s).any() else None for s in ['train','validation','test']},
        'warnings': [
            'No-record grid cells are NOT confirmed nonfires or valid sensor negatives.',
            'Satellite overpasses, cloud masking and viewing footprints are absent from FIRMS hotspot archives.',
            'Global VIIRS any-record indicator is not proven pixel/day satellite coverage.',
            'The 2024 S-NPP anomaly indicator covers known dates ONLY; other unverified outages are not encoded.',
            'Validation/test target dates are held out; their historical input windows may share earlier days, which is valid for next-day forecasting.',
            'This data supports a detection-proxy experiment only, not wildfire ignition prediction or operational accuracy.',
            'Background grid cells include areas outside the state polygon; never interpret background as validated fire-free area.',
        ],
    }
    (outdir/'dataset_card.json').write_text(json.dumps(card, indent=2))
    print('Detected NASA input files:', len(logs))
    for row in logs: print(' ', row['sensor'], row['source'], 'in-scope rows=', row['rows_in_date_and_bbox'])
    print('Unique observations:', counts)
    print('X:', X.shape, 'y:', y.shape)
    print('Splits (target dates):', card['split_target_date_ranges'])
    print('Output:', outdir.resolve())
    return card


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', required=True, help='Folder containing arbitrary-named CSV/ZIP NASA archive downloads.')
    ap.add_argument('--out', default='output/data')
    ap.add_argument('--start', default=DEFAULT_START)
    ap.add_argument('--end', default=DEFAULT_END)
    ap.add_argument('--km', type=int, default=10)
    ap.add_argument('--history', type=int, default=7)
    args = ap.parse_args()
    build(args.input_root, args.out, args.start, args.end, args.km, args.history)
