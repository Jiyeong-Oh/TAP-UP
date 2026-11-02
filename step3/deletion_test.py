"""Deletion test for the saliency maps."""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saliency_utils import REPO, load_model
from dataset import BCRDataset, MODALITIES
import saliency_methods
CKPT_BASE = f'{REPO}/checkpoints_stage2'
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]



def logits_from_volumes(model, vols, tumor_mask, tumor_present, cache, changed):
    b = next(iter(vols.values())).shape[0]
    (seqs, grid) = ([], None)
    for m in model.modalities:
        if m in changed:
            (tokens, x4) = model._branch_tokens(m, vols[m])
            grid = model._token_grid(x4)
        else:
            tokens = cache[m]['tokens'].expand(b, -1, -1)
            grid = cache[m]['grid']
        seqs.append(tokens)
    fused = model.fusion(seqs, grid, tumor_mask=tumor_mask.expand(b, -1, -1, -1, -1), tumor_present=tumor_present.expand(b))
    return model.head(fused).squeeze(-1).float()



def run(args):
    preds = pd.read_csv(f'{REPO}/data/predictions_{args.tag}.csv')
    manifest = pd.read_csv(f'{REPO}/data/bcr_manifest.csv')
    gland_paths = dict(zip(manifest[manifest['modality'] == 'gland']['patient_id'], manifest[manifest['modality'] == 'gland']['npy_path']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for fold in sorted(preds['fold'].unique()):
        ckpt = f'{CKPT_BASE}/{args.tag}/fold{fold}.pth'
        model = load_model(ckpt, MODALITIES)
        pids = preds[preds['fold'] == fold]['patient_id'].tolist()
        ds = BCRDataset(pids=pids, modalities=MODALITIES, load_tumor_mask=True, augment=False)
        need_mean = args.fill == 'mean' or args.saliency_method.endswith('_mean')
        mean_vol = saliency_methods.fold_mean_volumes(preds[preds['fold'] != fold]['patient_id'], device) if need_mean else None
        for (i, pid) in enumerate(pids):
            (volumes, present, label, _, info) = ds[i]
            vol = {m: v.unsqueeze(0).to(device) for (m, v) in volumes.items()}
            pres = {m: p_.unsqueeze(0).to(device) for (m, p_) in present.items()}
            tmask = info['mask'].unsqueeze(0).to(device)
            tpres = info['present'].unsqueeze(0).to(device)
            sal = saliency_methods.compute(args.saliency_method, model, vol, tmask, tpres, present=pres, baseline=mean_vol)
            gland = np.load(gland_paths[pid]) > 0.5
            gidx = np.flatnonzero(gland.ravel())
            vol_np = {m: vol[m][0, 0].cpu().numpy() for m in MODALITIES}
            blur = {m: gaussian_filter(vol_np[m], sigma=args.blur_sigma) for m in MODALITIES} if args.fill == 'blur' else {m: mean_vol[m][0, 0].cpu().numpy() for m in MODALITIES}
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                cache = {}
                for m in MODALITIES:
                    (t, x4) = model._branch_tokens(m, vol[m])
                    cache[m] = {'tokens': t, 'grid': model._token_grid(x4)}
                base_logit = float(logits_from_volumes(model, vol, tmask, tpres, cache, changed=())[0])
                rng = np.random.default_rng(args.seed)
                noise_orders = []
                for _ in range(args.n_random):
                    field = gaussian_filter(rng.standard_normal(gland.shape), sigma=args.blur_sigma).ravel()
                    noise_orders.append(gidx[np.argsort(-field[gidx])])
                sal_orders = {m: gidx[np.argsort(-sal[m].ravel()[gidx])] for m in MODALITIES}
                for cond in ('t2', 'adc', 'pet', 'all'):
                    changed = MODALITIES if cond == 'all' else (cond,)
                    for order_name in ['saliency'] + [f'random{r}' for r in range(args.n_random)]:
                        batch = {m: [] for m in changed}
                        for f in FRACTIONS:
                            k = int(round(f * len(gidx)))
                            for m in changed:
                                order = sal_orders[m] if order_name == 'saliency' else noise_orders[int(order_name[6:])]
                                v = vol_np[m].copy().ravel()
                                v[order[:k]] = blur[m].ravel()[order[:k]]
                                batch[m].append(v.reshape(vol_np[m].shape))
                        vb = {m: torch.from_numpy(np.stack(batch[m])[:, None]).to(device) for m in changed}
                        lg = logits_from_volumes(model, vb, tmask, tpres, cache, changed).cpu().numpy()
                        for (f, l) in zip(FRACTIONS, lg):
                            rows.append({'patient_id': pid, 'fold': fold, 'label': int(label), 'condition': cond, 'order': order_name, 'fraction': f, 'logit': float(l), 'base_logit': base_logit})
            print(f'fold {fold} {pid}: label={int(label)} base_logit={base_logit:.4f}', flush=True)
        del model
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(f'{args.out_dir}/deletion_curves.csv', index=False)
    print('max |logit(f=0) - base| =', (out[out.fraction == 0].logit - out[out.fraction == 0].base_logit).abs().max())



def summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    d = pd.read_csv(f'{args.out_dir}/deletion_curves.csv')
    d['dlogit'] = d['logit'] - d['base_logit']
    d['kind'] = np.where(d['order'] == 'saliency', 'saliency', 'random')
    curve = d.groupby(['patient_id', 'label', 'condition', 'kind', 'fraction'])['dlogit'].mean().reset_index()


    def area(g):
        g = g.sort_values('fraction')
        return getattr(np, 'trapezoid', np.trapz)(g['dlogit'], g['fraction'])
    aud = curve.groupby(['patient_id', 'label', 'condition', 'kind']).apply(area).rename('area').reset_index()
    rows = []
    for cond in ('t2', 'adc', 'pet', 'all'):
        for (grp, sel) in (('all patients', slice(None)), ('BCR+', 1)):
            a = aud[aud.condition == cond]
            if sel != slice(None):
                a = a[a.label == sel]
            w = a.pivot(index='patient_id', columns='kind', values='area')
            diff = w['saliency'] - w['random']
            p = wilcoxon(w['saliency'], w['random']).pvalue if len(w) > 5 else np.nan
            rows.append({'condition': cond, 'group': grp, 'n': len(w), 'area_saliency_mean': w['saliency'].mean(), 'area_random_mean': w['random'].mean(), 'diff_mean': diff.mean(), 'frac_patients_saliency_lower': (diff < 0).mean(), 'wilcoxon_p': p})
    t = pd.DataFrame(rows)
    t.to_csv(f'{args.out_dir}/deletion_summary.csv', index=False)
    print(t.round(4).to_string(index=False))
    names = {'t2': 'T2W only', 'adc': 'ADC only', 'pet': 'PSMA-PET only', 'all': 'All modalities'}
    (fig, axes) = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for (ax, cond) in zip(axes, ('t2', 'adc', 'pet', 'all')):
        for (kind, col) in (('saliency', '#d62728'), ('random', '#7f7f7f')):
            c = curve[(curve.condition == cond) & (curve.kind == kind)].groupby('fraction')['dlogit']
            (m, se) = (c.mean(), c.sem())
            ax.plot(m.index, m.values, '-o', color=col, ms=3, label=f'{kind}-ordered')
            ax.fill_between(m.index, m - se, m + se, color=col, alpha=0.2)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(names[cond])
        ax.set_xlabel('fraction of gland voxels removed')
    axes[0].set_ylabel('change in BCR logit (mean ± SE, n=84)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Deletion test: removing high-saliency vs random gland regions', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/deletion_curves.png', dpi=150)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--saliency_method', default='ig_mean', choices=['ig_mean'])
    p.add_argument('--fill', default='mean', choices=['blur', 'mean'])
    p.add_argument('--blur_sigma', type=float, default=4.0)
    p.add_argument('--n_random', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    summarize(args) if args.summarize else run(args)
if __name__ == '__main__':
    main()
