"""Overlap between modality saliency maps inside the gland: top-10 % Dice and Spearman."""
import argparse
import glob
import itertools
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MODS = ('t2', 'adc', 'pet')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    p.add_argument('--sigma_mm', type=float, default=1.5)
    p.add_argument('--top_pct', type=float, default=10.0)
    args = p.parse_args()
    rows = []
    for d in sorted(glob.glob(f'{REPO}/outputs/{args.tag}/nifti/*')):
        pid = os.path.basename(d)
        gimg = nib.load(f'{d}/{pid}_mask_gland.nii.gz')
        sp = gimg.header.get_zooms()[:3]
        g = (np.asarray(gimg.dataobj) > 0) | (np.asarray(nib.load(f'{d}/{pid}_mask_tumor.nii.gz').dataobj) > 0)
        sal = {m: gaussian_filter(np.asarray(nib.load(f'{d}/{pid}_saliency_raw_{m}.nii.gz').dataobj, dtype=np.float32),
                                  sigma=[args.sigma_mm / s for s in sp])[g] for m in MODS}
        k = max(1, int(round(g.sum() * args.top_pct / 100)))
        top = {m: set(np.argsort(v)[::-1][:k].tolist()) for m, v in sal.items()}
        for a, b in itertools.combinations(MODS, 2):
            rows.append({'patient_id': pid, 'pair': f'{a}-{b}', 'dice': 2 * len(top[a] & top[b]) / (2 * k),
                         'spearman': spearmanr(sal[a], sal[b]).correlation})
    df = pd.DataFrame(rows)
    df.to_csv(f'{REPO}/outputs/{args.tag}/saliency_overlap.csv', index=False)
    print(df.groupby('pair')[['dice', 'spearman']].median().round(3))


if __name__ == '__main__':
    main()
