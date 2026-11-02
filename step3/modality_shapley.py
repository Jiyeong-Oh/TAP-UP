"""Per-patient baseline Shapley values over the 2^3 modality subsets."""
import argparse
import itertools
import os
import sys
from math import factorial
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saliency_utils import REPO, load_model
from dataset import BCRDataset, MODALITIES
import saliency_methods
CKPT_BASE = f'{REPO}/checkpoints_stage2'



def reference_volumes(kind, train_pids, device):
    if kind == 'mean':
        return saliency_methods.fold_mean_volumes(train_pids, device)
    ds = BCRDataset(pids=list(train_pids), modalities=MODALITIES, load_tumor_mask=False, augment=False)
    if kind == 'zero':
        return {m: torch.zeros_like(x).unsqueeze(0).to(device) for (m, x) in ds[0][0].items()}
    vols = [ds[j][0] for j in range(len(ds))]
    return {m: torch.stack([v[m] for v in vols]).median(0).values.unsqueeze(0).to(device) for m in MODALITIES}



def shapley(v, players):
    n = len(players)
    phi = {}
    for p in players:
        rest = [q for q in players if q != p]
        s = 0.0
        for k in range(n):
            for S in itertools.combinations(rest, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                s += w * (v[frozenset(S) | {p}] - v[frozenset(S)])
        phi[p] = s
    return phi



def interaction(v, a, b, players):
    rest = [q for q in players if q not in (a, b)]
    (n, s) = (len(players), 0.0)
    for k in range(len(rest) + 1):
        for S in itertools.combinations(rest, k):
            S = frozenset(S)
            w = factorial(k) * factorial(n - k - 2) / factorial(n - 1)
            s += w * (v[S | {a, b}] - v[S | {a}] - v[S | {b}] + v[S])
    return s



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    p.add_argument('--reference', choices=['mean', 'median', 'zero'], default='mean')
    args = p.parse_args()
    device = torch.device('cuda')
    out_dir = f'{REPO}/outputs/{args.tag}'
    os.makedirs(out_dir, exist_ok=True)
    preds = pd.read_csv(f'{REPO}/data/predictions_{args.tag}.csv')
    rows = []
    for fold in sorted(preds['fold'].unique()):
        model = load_model(f'{CKPT_BASE}/{args.tag}/fold{fold}.pth', MODALITIES)
        model.eval()
        mean_vol = reference_volumes(args.reference, preds[preds['fold'] != fold]['patient_id'], device)
        pids = preds[preds['fold'] == fold]['patient_id'].tolist()
        ds = BCRDataset(pids=pids, modalities=MODALITIES, load_tumor_mask=True, augment=False)
        for (i, pid) in enumerate(pids):
            (volumes, _, label, _, info) = ds[i]
            vol = {m: x.unsqueeze(0).to(device) for (m, x) in volumes.items()}
            tmask = info['mask'].unsqueeze(0).to(device)
            tpres = info['present'].unsqueeze(0).to(device)
            v = {}
            with torch.no_grad():
                for k in range(len(MODALITIES) + 1):
                    for S in itertools.combinations(MODALITIES, k):
                        x = {m: vol[m] if m in S else mean_vol[m] for m in MODALITIES}
                        v[frozenset(S)] = saliency_methods._forward_logit(model, x, tmask, tpres).item()
            phi = shapley(v, list(MODALITIES))
            row = {'patient_id': pid, 'fold': fold, 'label': int(label), 'logit_full': v[frozenset(MODALITIES)], 'logit_none': v[frozenset()], **{f'phi_{m}': phi[m] for m in MODALITIES}, **{f'I_{a}_{b}': interaction(v, a, b, list(MODALITIES)) for (a, b) in itertools.combinations(MODALITIES, 2)}, **{'v_' + ('+'.join(sorted(S)) or 'none'): val for (S, val) in v.items()}}
            rows.append(row)
            print(f'fold {fold} {pid} done', flush=True)
        del model
        torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(f'{out_dir}/modality_shapley' + ('' if args.reference == 'mean' else f'_ref{args.reference}') + '.csv', index=False)
    phis = d[[f'phi_{m}' for m in MODALITIES]]
    share = phis.abs().div(phis.abs().sum(1), axis=0)
    q = lambda s: f'{s.median():.3f} ({s.quantile(0.25):.3f}-{s.quantile(0.75):.3f})'
    print(f'\n{len(d)} patients; efficiency check max |sum phi - (full - none)| = {(phis.sum(1) - (d.logit_full - d.logit_none)).abs().max():.2e}')
    print('phi (logit), median (IQR):')
    print(phis.agg(q).to_string())
    print('|phi| share, median:', share.median().round(2).to_dict())
    print('dominant modality (largest |phi|):', share.idxmax(axis=1).str[4:].value_counts().to_dict())
    print('dominant by BCR:')
    print(pd.crosstab(d.label, share.idxmax(axis=1).str[4:]))
    inter = d[[c for c in d.columns if c.startswith('I_')]]
    print('pairwise interaction, median (IQR):')
    print(inter.agg(q).to_string())
    print('fraction of patients with synergy (I > 0):', (inter > 0).mean().round(2).to_dict())
if __name__ == '__main__':
    main()
