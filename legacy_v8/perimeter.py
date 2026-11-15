"""CAL FIRE historical Park Fire perimeter retrieval and reproducible spatial filtering.
No hardcoded/improvised polygon. This geometry is a final historical footprint.
"""
from __future__ import annotations
import io
import json
import requests
import numpy as np
import pandas as pd
import pydeck as pdk
from shapely.geometry import Point, shape, mapping
from shapely.ops import transform
from shapely.prepared import prep
from pyproj import Transformer
from functools import lru_cache

SOURCE = 'https://services3.arcgis.com/CSK2qijkklTsqQyF/ArcGIS/rest/services/CaliforniaFirePerimeters2024/FeatureServer/0/query'
MAP_CRS = 'EPSG:4326'
METRIC_CRS = 'EPSG:3310'  # California Albers, meter-based for small buffers
TO_METERS = Transformer.from_crs(MAP_CRS, METRIC_CRS, always_xy=True).transform
TO_WGS84 = Transformer.from_crs(METRIC_CRS, MAP_CRS, always_xy=True).transform
PARK_POINT=Point(-121.76168,39.7789)

def _park_candidates(geojson):
    features=geojson.get('features',[]) if geojson.get('type')=='FeatureCollection' else [geojson] if geojson.get('type')=='Feature' else []
    matches=[]
    for f in features:
        attrs=f.get('properties',{}) or {}
        name=str(attrs.get('FIRE_NAME') or attrs.get('fire_name') or '').upper().strip()
        try: year=int(attrs.get('YEAR_') or attrs.get('year') or 0)
        except (ValueError,TypeError): year=0
        if 'PARK' not in name or year != 2024 or not f.get('geometry'):
            continue
        geom=shape(f['geometry'])
        if geom.is_empty or not geom.is_valid:
            geom=geom.buffer(0)
        if geom.is_empty or geom.geom_type not in ('Polygon','MultiPolygon'):
            continue
        near_km=transform(TO_METERS,geom).distance(transform(TO_METERS,PARK_POINT))/1000
        # Prevent another 'Park' fire elsewhere in California being selected.
        if near_km > 25:
            continue
        matches.append((near_km,geom,attrs))
    matches.sort(key=lambda p:p[0])
    return matches

def _select(geojson):
    candidates=_park_candidates(geojson)
    if not candidates: raise ValueError('No validated 2024 PARK fire polygon within 25 km of the Park Fire ignition location.')
    distance,geom,attrs=candidates[0]
    description=f'{attrs.get("FIRE_NAME", "Park")} · {attrs.get("YEAR_", 2024)} · GIS acres: {attrs.get("GIS_ACRES", "not listed")} · CAL FIRE FRAP historical footprint'
    return geom, description

@lru_cache(maxsize=2)
def fetch_official_park_perimeter():
    params={'where':"YEAR_ = 2024 AND FIRE_NAME LIKE '%PARK%'",'outFields':'OBJECTID,FIRE_NAME,YEAR_,GIS_ACRES,ALARM_DATE,CONT_DATE',
            'returnGeometry':'true','outSR':'4326','f':'geojson'}
    r=requests.get(SOURCE,params=params,timeout=30)
    r.raise_for_status()
    data=r.json()
    if 'error' in data: raise ValueError(f'ArcGIS service error: {data["error"]}')
    return _select(data)

def read_perimeter_geojson(blob:bytes):
    doc=json.loads(blob)
    return _select(doc)

def filter_perimeter_points(df:pd.DataFrame, polygon, buffer_km=0):
    polygon_metric=transform(TO_METERS,polygon)
    study=transform(TO_WGS84,polygon_metric.buffer(float(buffer_km)*1000)) if buffer_km else polygon
    exact=prep(polygon); buffered=prep(study)
    candidates=df.copy()
    # Bounding-box prefilter prevents checking the whole state against the complex polygon.
    minx,miny,maxx,maxy=study.bounds
    possible=candidates.longitude.between(minx,maxx)&candidates.latitude.between(miny,maxy)
    inside=np.zeros(len(candidates),dtype=bool)
    footprint=np.zeros(len(candidates),dtype=bool)
    for idx in np.flatnonzero(possible.to_numpy()):
        point=Point(float(candidates.iloc[idx].longitude),float(candidates.iloc[idx].latitude))
        inside[idx]=buffered.covers(point)
        footprint[idx]=exact.covers(point)
    return candidates.iloc[np.flatnonzero(inside)].copy(), {'inside_unbuffered':int(footprint.sum())}

def perimeter_geojson_bytes(polygon, description):
    obj={'type':'FeatureCollection','features':[{'type':'Feature', 'properties':{'FIRE_NAME':'PARK', 'YEAR_':2024, 'name':'Park Fire','source':'CAL FIRE FRAP historical perimeters', 'description':description,'caution':'Retrospective final footprint, not July 2024 daily perimeter'},'geometry':mapping(polygon)}]}
    return json.dumps(obj).encode('utf-8')

def perimeter_deck(poly,points):
    # pydeck PolygonLayer supports individual polygons; flatten MultiPolygons and holes aren't shown
    polys=list(poly.geoms) if poly.geom_type=='MultiPolygon' else [poly]
    # Hollow red outlines; holes excluded by spatial filter, despite outline-only visual.
    layer_poly=pdk.Layer('PolygonLayer',data=[{'polygon':list(g.exterior.coords)} for g in polys],get_polygon='polygon',
                         get_fill_color=[200,40,40,10],get_line_color=[200,35,35,240],get_line_width=3,
                         stroked=True,filled=True,wireframe=False,pickable=False)
    subset=points.sample(min(len(points),3000),random_state=12) if len(points)>3000 else points
    layer_points=pdk.Layer('ScatterplotLayer',data=subset[['longitude','latitude','sensor_group']].to_dict('records'),
                           get_position='[longitude, latitude]',get_radius=180,radius_min_pixels=2,pickable=True,
                           get_fill_color='sensor_group == "MODIS" ? [40,110,220,180] : [240,165,40,180]')
    center=poly.centroid
    return pdk.Deck(layers=[layer_poly,layer_points],initial_view_state=pdk.ViewState(latitude=center.y,longitude=center.x,zoom=8.3),
                    tooltip={'text':'{sensor_group}'},map_style='light')
