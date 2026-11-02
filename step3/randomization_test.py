"""Model randomization test for the saliency maps."""
import argparse
import copy
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saliency_utils import REPO, load_model
from dataset import BCRDataset, MODALITIES
import saliency_methods
CKPT_BASE = f'{REPO}/checkpoints_stage2'



def randomised_copy(model, seed=0):
    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    for mod in (rnd.fusion, rnd.head):
        for layer in mod.modules():
            if isinstance(layer, torch.nn.MultiheadAttention):
                layer._reset_parameters()
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    for p in rnd.missing_embed.values():
        torch.nn.init.normal_(p, std=0.02)
    return rnd.eval()



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--methods', default='ig_mean')
    args = p.parse_args()
    methods = args.methods.split(',')
    preds = pd.read_csv(f'{REPO}/data/predictions_{args.tag}.csv')
    manifest = pd.read_csv(f'{REPO}/data/bcr_manifest.csv')
    gland_paths = dict(zip(manifest[manifest['modality'] == 'gland']['patient_id'], manifest[manifest['modality'] == 'gland']['npy_path']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for fold in sorted(preds['fold'].unique()):
        ckpt = f'{CKPT_BASE}/{args.tag}/fold{fold}.pth'
        model = load_model(ckpt, MODALITIES)
        rnd = randomised_copy(model, seed=fold)
        mean_vol = saliency_methods.fold_mean_volumes(preds[preds['fold'] != fold]['patient_id'], device) if any((m.endswith('_mean') for m in methods)) else None
        pids = preds[preds['fold'] == fold]['patient_id'].tolist()
        ds = BCRDataset(pids=pids, modalities=MODALITIES, load_tumor_mask=True, augment=False)
        for (i, pid) in enumerate(pids):
            (volumes, present, label, _, info) = ds[i]
            vol = {m: v.unsqueeze(0).to(device) for (m, v) in volumes.items()}
            pres = {m: p_.unsqueeze(0).to(device) for (m, p_) in present.items()}
            tmask = info['mask'].unsqueeze(0).to(device)
            tpres = info['present'].unsqueeze(0).to(device)
            g = (np.load(gland_paths[pid]) > 0.5).ravel()
            for meth in methods:
                a = saliency_methods.compute(meth, model, vol, tmask, tpres, present=pres, baseline=mean_vol)
                b = saliency_methods.compute(meth, rnd, vol, tmask, tpres, present=pres, baseline=mean_vol)
                torch.cuda.empty_cache()
                for m in MODALITIES:
                    (x, y) = (a[m].ravel()[g], b[m].ravel()[g])
                    ok = x.std() > 0 and y.std() > 0
                    rows.append({'patient_id': pid, 'fold': fold, 'label': int(label), 'method': meth, 'modality': m, 'pearson': pearsonr(x, y)[0] if ok else np.nan, 'spearman': spearmanr(x, y)[0] if ok else np.nan})
            print(f'fold {fold} {pid} done', flush=True)
        del model, rnd
        torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(f'{args.out_dir}/randomization_corr.csv', index=False)
    s = d.groupby(['method', 'modality'])[['pearson', 'spearman']].median().round(3)
    s.to_csv(f'{args.out_dir}/randomization_summary.csv')
    print(s.to_string())
if __name__ == '__main__':
    main()
