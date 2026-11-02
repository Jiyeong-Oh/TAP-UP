"""Held-out integrated-gradients maps of one fold, exported as NIfTI.

For each patient: images, masks, the positive IG map (saliency_raw_<m>) and its
display version (1.5 mm Gaussian, scaled to the 99th gland percentile, saliency_<m>).
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saliency_utils import REPO, gland_and_spacing, load_model
from dataset import BCRDataset, MODALITIES
import saliency_methods


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--sigma_mm', type=float, default=1.5)
    p.add_argument('--pct', type=float, default=99.0)
    args = p.parse_args()
    preds = pd.read_csv(f'{REPO}/data/predictions_{args.tag}.csv', dtype={'patient_id': str})
    pids = preds[preds['fold'] == args.fold]['patient_id'].tolist()
    model = load_model(f'{REPO}/checkpoints_stage2/{args.tag}/fold{args.fold}.pth', MODALITIES)
    device = next(model.parameters()).device
    baseline = saliency_methods.fold_mean_volumes(preds[preds['fold'] != args.fold]['patient_id'], device)
    ds = BCRDataset(pids, MODALITIES, load_tumor_mask=True)
    for i, pid in enumerate(pids):
        volumes, present, _, _, info = ds[i]
        volumes = {m: v.unsqueeze(0).to(device) for m, v in volumes.items()}
        present = {m: v.unsqueeze(0).to(device) for m, v in present.items()}
        tumor = info['mask'].unsqueeze(0).to(device)
        raw = saliency_methods.compute('ig_mean', model, volumes, tumor, info['present'].unsqueeze(0).to(device),
                                       present=present, baseline=baseline, raw=True)
        gland, spacing = gland_and_spacing(pid)
        out = f'{REPO}/outputs/{args.tag}/nifti/{pid}'
        os.makedirs(out, exist_ok=True)
        affine = np.diag([*spacing, 1.0])

        def save(arr, name, dtype=np.float32):
            nib.save(nib.Nifti1Image(np.asarray(arr).astype(dtype), affine), f'{out}/{pid}_{name}.nii.gz')

        for m in MODALITIES:
            positive = np.clip(raw[m], 0, None)
            smooth = gaussian_filter(positive, sigma=[args.sigma_mm / s for s in spacing])
            save(volumes[m][0, 0].detach().cpu().numpy(), f'image_{m}')
            save(positive, f'saliency_raw_{m}')
            save(np.clip(smooth / (np.percentile(smooth[gland > 0.5], args.pct) + 1e-12), 0, 1), f'saliency_{m}')
        save(gland > 0.5, 'mask_gland', np.uint8)
        save(tumor[0, 0].cpu().numpy() > 0.5, 'mask_tumor', np.uint8)
        print(pid, 'done', flush=True)


if __name__ == '__main__':
    main()
