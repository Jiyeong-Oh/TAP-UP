"""Preprocess the pretraining volumes and cache them as .npy with a manifest."""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_sources as ds
from crop_utils import crop_around_mask, resample_to_isotropic, resize_to_shape
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CACHE_DIR = os.environ.get('TAPUP_CACHE', 'cache') + '/ssl'
MANIFEST_OUT = f'{REPO}/data/ssl_manifest.csv'
MARGIN_MM = 3.0
TARGET_SPACING = 0.5
TARGET_SHAPE = (128, 128, 64)



def percentile_norm_in_mask(vol, mask, lo=1, hi=99):
    vals = vol[mask > 0.5]
    if vals.size == 0:
        vals = vol.ravel()
    (p_lo, p_hi) = np.percentile(vals, [lo, hi])
    p_hi = max(p_hi, p_lo + 1e-06)
    return np.clip((vol - p_lo) / (p_hi - p_lo), 0, 1)



def preprocess_one(vol, gland, spacing):
    out = crop_around_mask(vol, gland, margin_mm=MARGIN_MM, spacing=spacing)
    if out is None:
        return None
    (vol_c, gland_c, _) = out
    vol_rs = resample_to_isotropic(vol_c, spacing, TARGET_SPACING, order=1)
    gland_rs = resample_to_isotropic(gland_c, spacing, TARGET_SPACING, order=0)
    vol_rz = resize_to_shape(vol_rs, TARGET_SHAPE, order=1)
    gland_rz = resize_to_shape(gland_rs, TARGET_SHAPE, order=0)
    return percentile_norm_in_mask(vol_rz, gland_rz).astype(np.float32)



def preprocess_mask(mask_vol, gland, spacing):
    out = crop_around_mask(mask_vol, gland, margin_mm=MARGIN_MM, spacing=spacing)
    if out is None:
        return None
    (mask_c, _, _) = out
    mask_rs = resample_to_isotropic(mask_c, spacing, TARGET_SPACING, order=0)
    mask_rz = resize_to_shape(mask_rs, TARGET_SHAPE, order=0)
    return (mask_rz > 0.5).astype(np.float32)



def process_patient(source, pid, data, rows):
    for mod in ('t2', 'adc', 'pet'):
        vol = data.get(mod)
        if vol is None:
            continue
        out_dir = f'{CACHE_DIR}/{source}/{mod}'
        out_path = f'{out_dir}/{pid}.npy'
        if os.path.exists(out_path):
            rows.append({'source': source, 'patient_id': pid, 'modality': mod, 'npy_path': out_path})
            continue
        out = preprocess_one(vol, data['gland'], data['spacing'])
        if out is None:
            print(f'{source}/{pid}/{mod}: SKIP (empty gland mask)')
            continue
        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, out)
        rows.append({'source': source, 'patient_id': pid, 'modality': mod, 'npy_path': out_path})
    for mask_name in ('gland', 'tumor'):
        mask_vol = data.get(mask_name)
        if mask_vol is None:
            continue
        out_dir = f'{CACHE_DIR}/{source}/{mask_name}'
        out_path = f'{out_dir}/{pid}.npy'
        if os.path.exists(out_path):
            rows.append({'source': source, 'patient_id': pid, 'modality': mask_name, 'npy_path': out_path})
            continue
        out = preprocess_mask(mask_vol, data['gland'], data['spacing'])
        if out is None:
            print(f'{source}/{pid}/{mask_name}: SKIP (empty gland mask)')
            continue
        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, out)
        rows.append({'source': source, 'patient_id': pid, 'modality': mask_name, 'npy_path': out_path})



def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    for (source, loader, pids) in [('internal_unlabeled', ds.load_internal, ds.list_patients('internal_unlabeled')), ('picai', ds.load_picai, ds.list_patients('picai')), ('deeppsma_psma', ds.load_deeppsma_psma_pet, ds.list_deeppsma_usable())]:
        print(f'=== {source}: {len(pids)} patients ===')
        for pid in pids:
            try:
                process_patient(source, pid, loader(pid), rows)
            except Exception as e:
                print(f'{source}/{pid}: FAILED {repr(e)}')
    for split_name in ('train', 'test'):
        source = f'prostate158_{split_name}'
        pids = ds.list_patients(source)
        print(f'=== {source}: {len(pids)} patients ===')
        for pid in pids:
            try:
                process_patient(source, pid, ds.load_prostate158(pid, split=split_name), rows)
            except Exception as e:
                print(f'{source}/{pid}: FAILED {repr(e)}')
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_OUT, index=False)
    print(f'wrote {MANIFEST_OUT}: {len(df)} rows')
    print(df.groupby(['source', 'modality']).size())
if __name__ == '__main__':
    main()
