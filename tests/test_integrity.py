"""Independent numerical/provenance checks; run python tests/test_integrity.py"""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1]
source=root/'data'/'predict';hist=root/'data'/'historic'
manifest=pd.read_csv(source/'sequence_manifest.csv')
assert len(manifest)==1454
assert manifest.split.value_counts().to_dict()=={'train':870,'validation':292,'test':292}
assert tuple(manifest[manifest.split=='validation'].target_date_utc.iloc[[0,-1]])==('2024-05-27','2025-03-14')
assert tuple(manifest[manifest.split=='test'].target_date_utc.iloc[[0,-1]])==('2025-03-15','2025-12-31')
z=np.load(source/'test_predictions.npz',allow_pickle=False)
scores=z['predicted_detection_proxy_scores']; actual=z['actual_recorded_hotspot_labels'].astype(bool)
assert scores.shape==(292,99,103) and actual.shape==(292,99,103)
assert list(z['target_dates'])==list(manifest[manifest.split=='test'].target_date_utc)
threshold=json.loads((source/'model_card.json').read_text())['selected_threshold_on_validation_only']
pred=scores>=threshold
counts=(int((pred&actual).sum()),int((pred&~actual).sum()),int((~pred&actual).sum()))
assert counts==(3581,4363,5480),counts
m=pd.read_csv(source/'model_metrics.csv')
t=m[m.subset=='test'].iloc[0];assert all(int(t[k])==v for k,v in zip(['tp','fp','fn'],counts))
b=pd.read_csv(source/'baseline_metrics.csv')
assert (b[(b.subset=='test')&(b.model=='Persistence')][['tp','fp','fn']].iloc[0].astype(int).tolist())==[3460,5603,5601]
park=pd.read_csv(hist/'park_evidence_groups.csv')
assert len(park)==316
assert (park['Both sensors have hotspot records'].sum(),park['At least one time-matched pair'].sum())==(54,45)
assert pd.read_csv(hist/'park_hotspot_records.csv').shape[0]==3092
assert len(pd.read_csv(hist/'california_july_2024_hotspots.csv'))==12621
assert sum((hist/f).exists() for f in ['fire_archive_M-C61_850130.csv','fire_archive_SV-C2_850131.csv'])==2
print('PASS: date split, arrays, model score, baseline, raw detections, V8 Park evidence.')
