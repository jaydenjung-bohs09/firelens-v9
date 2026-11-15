#!/usr/bin/env python3
"""Small ConvLSTM against persistence/neighborhood detection-proxy baselines.

Science limitation: FIRMS hotspot archives contain no verified negative labels.
Metrics here indicate overlap with RECORDS of next-day hotspot positives and
must never be described as wildfire accuracy, sensitivity, or ignition forecasts.
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X, self.y, self.indices = X, y, np.asarray(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        # (time,height,width,channels) to (time,channels,height,width)
        x = np.transpose(self.X[idx], (0,3,1,2)).copy()
        return torch.from_numpy(x).float(), torch.from_numpy(self.y[idx].copy()).float()


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4*hidden_dim, kernel_size=3, padding=1)
    def forward(self, x, h, c):
        i, f, g, o = self.gates(torch.cat([x, h], 1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f*c + i*torch.tanh(g)
        h = o*torch.tanh(c)
        return h, c


class SmallConvLSTM(nn.Module):
    def __init__(self, channels=4, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(channels, hidden_dim)
        self.head = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_dim, 1, 1))
    def forward(self, x):
        batch, t, _, height, width = x.shape
        h = x.new_zeros((batch, self.hidden_dim, height, width))
        c = x.new_zeros((batch, self.hidden_dim, height, width))
        for step in range(t):
            h, c = self.cell(x[:, step], h, c)
        return self.head(h).squeeze(1)


def counts(pred, target):
    pred, target = np.asarray(pred).astype(bool), np.asarray(target).astype(bool)
    return (int((pred&target).sum()), int((pred&~target).sum()), int((~pred&target).sum()))


def metrics_from_counts(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                precision=float(tp/(tp+fp)) if tp+fp else 0.0,
                recall=float(tp/(tp+fn)) if tp+fn else 0.0,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
                iou=float(tp/(tp+fp+fn)) if tp+fp+fn else 0.0)


def score_binary(pred, target):
    return metrics_from_counts(*counts(pred, target))


def evaluate_baselines(X, y, idx, subset):
    rows = []
    if not len(idx): return rows
    last = X[idx, -1, :, :, :2].max(axis=-1).astype(bool)
    neighborhood = binary_dilation(last, structure=np.ones((1,3,3), dtype=bool))
    for name, pred in [('Persistence',last), ('3x3-neighborhood persistence', neighborhood)]:
        rows.append(dict(subset=subset, model=name, **score_binary(pred, y[idx])))
    return rows


def infer_probabilities(model, loader, device):
    model.eval()
    outs=[]
    with torch.inference_mode():
        for xb,_ in loader:
            xb=xb.to(device)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy().astype(np.float16))
    return np.concatenate(outs, axis=0)


def train(npz_path, manifest_path, outdir, epochs=5, batch_size=8, hidden=8,
          lr=0.001, max_train_samples=0, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # Disable excessive CPU oversubscription (Colab CPU safety).
    if not torch.cuda.is_available(): torch.set_num_threads(min(torch.get_num_threads(), 4))
    d = np.load(npz_path)
    X, y, dates = d['X'], d['y'], d['dates']
    m = pd.read_csv(manifest_path)
    if len(m) != len(y) or not np.array_equal(m.target_date_utc.astype(str).to_numpy(), dates.astype(str)):
        raise ValueError('Manifest and sequence NPZ differ; rebuild both using the same run.')
    indices = {split: np.flatnonzero(m['split'].to_numpy()==split) for split in ['train','validation','test']}
    if any(len(v)==0 for v in indices.values()):
        raise ValueError(f'Empty train/validation/test split: { {k:len(v) for k,v in indices.items()} }')
    if max_train_samples:
        indices['train'] = indices['train'][:max_train_samples]
    if y.shape[0] < 50:
        print('WARNING: Tiny dataset; model training is a code smoke test, not a validation of forecasting.')
    if y[indices['train']].sum() == 0:
        raise ValueError('No recorded positive target cells in the training set; cannot train a meaningful proxy model.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    dataloaders = {split: DataLoader(SequenceDataset(X, y, ids), batch_size=batch_size, shuffle=(split=='train'), num_workers=0)
                   for split,ids in indices.items()}
    network = SmallConvLSTM(channels=X.shape[-1], hidden_dim=hidden).to(device)
    opt = torch.optim.Adam(network.parameters(), lr=lr)
    positives = int(y[indices['train']].sum())
    negatives = int(y[indices['train']].size - positives)
    raw_ratio = negatives/max(positives, 1)
    # Cap imbalance weight; predictions are NOT calibrated probabilities.
    weight = 10.0
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    print('Device:',device, 'positive proxy cell prevalence:', round(positives / y[indices['train']].size, 6), 'loss_pos_weight:', weight)
    print('Split sizes:', {k:len(v) for k,v in indices.items()})
    history = []; best = float('inf'); best_state = None; patience=5; stale=0
    for epoch in range(1, epochs+1):
        summary={}
        for split in ['train','validation']:
            network.train(split=='train')
            numerator=0.0; denominator=0
            for xb,yb in dataloaders[split]:
                xb,yb=xb.to(device),yb.to(device)
                with torch.set_grad_enabled(split=='train'):
                    logits=network(xb)
                    loss=lossfn(logits,yb)
                    if split=='train':
                        opt.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(),1.0)
                        opt.step()
                numerator += loss.item()*len(xb); denominator += len(xb)
            summary[split] = numerator/max(denominator,1)
        history.append({'epoch':epoch, 'train_loss':summary['train'], 'val_loss':summary['validation']})
        print(f'Epoch {epoch}/{epochs}: train_loss={summary["train"]:.6f}, val_loss={summary["validation"]:.6f}',flush=True)
        if summary['validation'] < best-1e-6:
            best = summary['validation']; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print('Early stopping on validation loss.')
                break
    network.load_state_dict(best_state)
    torch.save({'model_state_dict':best_state, 'channels':int(X.shape[-1]),'hidden_dim':hidden,'history_days':int(X.shape[1]),
                'note':'Experimental next-day detected-hotspot proxy only; NOT wildfire occurrence probability.'},outdir/'convlstm_weights.pt')
    pd.DataFrame(history).to_csv(outdir/'training_history.csv',index=False)

    baseline_rows = evaluate_baselines(X,y,indices['validation'],'validation') + evaluate_baselines(X,y,indices['test'],'test')
    pd.DataFrame(baseline_rows).to_csv(outdir/'baseline_metrics.csv',index=False)
    val_probs = infer_probabilities(network,dataloaders['validation'],device)
    test_probs = infer_probabilities(network,dataloaders['test'],device)
    val_target = y[indices['validation']]
    test_target = y[indices['test']]
    thresholds = [round(float(v), 3) for v in np.arange(0.05, 0.951, 0.005)]
    threshold_sweep = [dict(threshold=threshold, **score_binary(val_probs >= threshold, val_target)) for threshold in thresholds]
    pd.DataFrame(threshold_sweep).to_csv(outdir/'validation_threshold_sweep.csv',index=False)
    best_threshold = sorted(threshold_sweep,key=lambda row:(-row['f1'],row['threshold']))[0]['threshold']
    model_rows = [dict(subset='validation',model='ConvLSTM',threshold=best_threshold,
                       **score_binary(val_probs>=best_threshold,val_target)),
                  dict(subset='test',model='ConvLSTM',threshold=best_threshold,
                       **score_binary(test_probs>=best_threshold,test_target))]
    pd.DataFrame(model_rows).to_csv(outdir/'model_metrics.csv',index=False)
    np.savez_compressed(outdir/'test_predictions.npz', predicted_detection_proxy_scores=test_probs,
                        actual_recorded_hotspot_labels=test_target,
                        target_dates=dates[indices['test']])
    card = {'task':'next-day presence of positive MODIS-or-VIIRS S-NPP hotspot records, NOT ignition/real fire absence',
            'architecture':'Single ConvLSTM (hidden 8 default) + convolutional output head',
            'model_prediction_type':'uncalibrated detection-proxy score', 'device':str(device),
            'split_sizes':{k:len(v) for k,v in indices.items()},
            'best_epoch':int(np.argmin([row['val_loss'] for row in history])+1),
            'selected_threshold_on_validation_only':best_threshold,
            'positive_label_cell_prevalence_in_train':positives / y[indices['train']].size,
            'positive_loss_weight':weight,'max_train_samples':max_train_samples,
            'remarks':['No-hotspot pixels are unlabeled / not observed positives, not verified negatives.',
                       'Precision/recall/F1/IoU measure proxy hotspot-record overlap and cannot establish fire detection accuracy.',
                       'The final test period was not used to tune the threshold or early stopping.',
                       'Known S-NPP anomalies must be disclosed; see sequence_manifest.csv for target-day flags.',
                       'Positive grids from one fire may persist many days; temporal test cannot prove independent-event generalization.']}
    (outdir/'model_card.json').write_text(json.dumps(card,indent=2))
    print('Baselines:\n',pd.DataFrame(baseline_rows).to_string(index=False))
    print('ConvLSTM:\n',pd.DataFrame(model_rows).to_string(index=False))
    print('Uncalibrated proxy score threshold selected using validation only:',best_threshold)
    print('Saved:',outdir.resolve())
    return card


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--data',default='output/data/firelens_sequences.npz')
    a.add_argument('--manifest',default='output/data/sequence_manifest.csv')
    a.add_argument('--out',default='output/training')
    a.add_argument('--epochs',type=int,default=5)
    a.add_argument('--batch-size',type=int,default=8)
    a.add_argument('--hidden',type=int,default=8)
    a.add_argument('--lr',type=float,default=0.001)
    a.add_argument('--max-train-samples',type=int,default=0,
                   help='For a quick test only; 0 uses the complete training split.')
    a.add_argument('--seed',type=int,default=42)
    args=a.parse_args()
    train(args.data,args.manifest,args.out,args.epochs,args.batch_size,args.hidden,args.lr,args.max_train_samples,args.seed)
