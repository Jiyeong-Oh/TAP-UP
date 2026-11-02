"""Fig. 2: modality saliency maps and the top-10 % voxels of each modality.

--pids takes the patients to show, one per row, in the form id:row_label.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saliency_utils import REPO, heatmap_rgba, outline

MODS = [('t2', 'T2W'), ('adc', 'ADC'), ('pet', 'PSMA-PET')]
TOP = {'t2': (0.85, 0.2, 0.85), 'adc': (0.2, 0.85, 0.3), 'pet': (1.0, 0.6, 0.1)}


def gray(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    p.add_argument('--pids', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--top_pct', type=float, default=10.0)
    args = p.parse_args()
    pr = pd.read_csv(f'{REPO}/data/predictions_{args.tag}.csv', dtype={'patient_id': str})
    fpr, tpr, thr = roc_curve(pr.label, pr.prob)
    prob = dict(zip(pr.patient_id, pr.prob))
    rows = [x.split(':') for x in args.pids.split(',')]
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.8, 2.25 * len(rows)), squeeze=False)
    for r, (pid, role) in enumerate(rows):
        load = lambda n: np.asarray(nib.load(f'{REPO}/outputs/{args.tag}/nifti/{pid}/{pid}_{n}.nii.gz').dataobj, dtype=np.float32)
        tumor, gland = load('mask_tumor'), load('mask_gland')
        z = int(np.argmax(tumor.sum(axis=(0, 1)))) if tumor.sum() > 0 else int(np.argmax(gland.sum(axis=(0, 1))))
        sal = {m: load(f'saliency_{m}') for m, _ in MODS}
        for c, (m, title) in enumerate(MODS):
            ax = axes[r, c]
            ax.imshow(gray(load(f'image_{m}')[:, :, z].T), cmap='gray')
            ax.imshow(heatmap_rgba(sal[m][:, :, z].T))
            ax.imshow(outline(gland[:, :, z].T))
            ax.imshow(outline(tumor[:, :, z].T, color=(0, 1, 1)))
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{role}\np = {prob[pid]:.3f}', fontsize=9)
        g3 = (gland > 0.5) | (tumor > 0.5)
        rgb = np.repeat(gray(load('image_t2')[:, :, z].T)[..., None], 3, axis=-1) * 0.6
        acc, cnt = np.zeros_like(rgb), np.zeros(rgb.shape[:2])
        for m, _ in MODS:
            hit = ((sal[m] >= np.percentile(sal[m][g3], 100 - args.top_pct)) & g3)[:, :, z].T
            acc[hit] += TOP[m]
            cnt[hit] += 1
        hit = cnt > 0
        rgb[hit] = 0.25 * rgb[hit] + 0.75 * acc[hit] / cnt[hit, None]
        ax = axes[r, 3]
        ax.imshow(rgb)
        ax.imshow(outline(gland[:, :, z].T))
        ax.imshow(outline(tumor[:, :, z].T, color=(0, 1, 1)))
        if r == 0:
            ax.set_title(f'Top {args.top_pct:.0f}% per modality', fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=TOP[m], label=n) for m, n in MODS], loc='lower right', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(pad=0.3, rect=(0, 0.035, 1, 1))
    fig.savefig(args.out, dpi=300)
    print('wrote', args.out, 'threshold', thr[np.argmax(tpr - fpr)])


if __name__ == '__main__':
    main()
