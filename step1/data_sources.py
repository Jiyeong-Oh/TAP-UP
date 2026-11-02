"""Loaders for the pretraining datasets and the paired cohort.

Dataset locations are read from environment variables; see the README for the
expected folder layouts. Each loader returns a dict with t2, adc, pet, gland,
tumor (arrays or None) and spacing.
"""
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ROOT_INTERNAL = os.environ.get('TAPUP_INTERNAL', 'datasets/paired')
ROOT_PICAI = os.environ.get('TAPUP_PICAI', 'datasets/picai')
ROOT_P158_TRAIN = os.environ.get('TAPUP_P158_TRAIN', 'datasets/prostate158/train')
ROOT_P158_TEST = os.environ.get('TAPUP_P158_TEST', 'datasets/prostate158/test')
ROOT_DEEPPSMA = os.environ.get('TAPUP_DEEPPSMA', 'datasets/deep-psma')
UNLABELED_CSV = f'{REPO}/data/internal_unlabeled.csv'
DEEPPSMA_PROSTATE_LABEL = 22


def _f32(path):
    return nib.load(path).get_fdata().astype(np.float32)


def _f32_or_none(path):
    return _f32(path) if os.path.exists(path) else None


def load_internal(pid):
    base = f'{ROOT_INTERNAL}/{pid}'
    t2_img = nib.load(f'{base}/t2w.nii.gz')
    return {'t2': t2_img.get_fdata().astype(np.float32), 'adc': _f32(f'{base}/adc.nii.gz'),
            'pet': _f32_or_none(f'{base}/pet.nii.gz'), 'gland': _f32(f'{base}/gland.nii.gz'),
            'tumor': _f32_or_none(f'{base}/tumor.nii.gz'), 'spacing': t2_img.header.get_zooms()[:3]}


def load_deeppsma_psma_pet(case_id):
    base = f'{ROOT_DEEPPSMA}/{case_id}/PSMA'
    pet_img = nib.load(f'{base}/PET.nii.gz')
    pet = pet_img.get_fdata().astype(np.float32)
    gland = (nib.load(f'{base}/totseg_24.nii.gz').get_fdata() == DEEPPSMA_PROSTATE_LABEL).astype(np.float32)
    if gland.shape != pet.shape:
        gland = zoom(gland, [p / c for p, c in zip(pet.shape, gland.shape)], order=0)
    return {'t2': None, 'adc': None, 'pet': pet, 'gland': gland, 'tumor': None,
            'spacing': pet_img.header.get_zooms()[:3]}


def list_deeppsma_usable():
    """Cases with a detectable prostate on the PSMA scan."""
    cases = []
    for case_id in sorted(os.listdir(ROOT_DEEPPSMA)):
        seg = f'{ROOT_DEEPPSMA}/{case_id}/PSMA/totseg_24.nii.gz'
        if os.path.exists(seg) and (nib.load(seg).get_fdata() == DEEPPSMA_PROSTATE_LABEL).any():
            cases.append(case_id)
    return cases


def load_picai(pid):
    base = f'{ROOT_PICAI}/{pid}/{pid}'
    t2_img = nib.load(f'{base}_t2w.nii.gz')
    return {'t2': t2_img.get_fdata().astype(np.float32), 'adc': _f32(f'{base}_adc_reg.nii.gz'), 'pet': None,
            'gland': _f32(f'{base}_gland.nii.gz'), 'tumor': _f32(f'{base}_tumor.nii.gz'),
            'spacing': t2_img.header.get_zooms()[:3]}


def load_prostate158(pid, split='train'):
    base = f'{ROOT_P158_TRAIN if split == "train" else ROOT_P158_TEST}/{pid}'
    t2_img = nib.load(f'{base}/t2.nii.gz')
    return {'t2': t2_img.get_fdata().astype(np.float32), 'adc': _f32(f'{base}/adc.nii.gz'), 'pet': None,
            'gland': (_f32(f'{base}/t2_anatomy_reader1.nii.gz') > 0.5).astype(np.float32),
            'tumor': _f32_or_none(f'{base}/t2_tumor_reader1.nii.gz'), 'spacing': t2_img.header.get_zooms()[:3]}


def list_patients(source):
    if source == 'internal_unlabeled':
        if not os.path.exists(UNLABELED_CSV):
            return []
        import pandas as pd
        return pd.read_csv(UNLABELED_CSV, dtype=str)['patient_id'].tolist()
    if source == 'picai':
        return sorted(os.listdir(ROOT_PICAI))
    if source == 'prostate158_train':
        return sorted(os.listdir(ROOT_P158_TRAIN))
    if source == 'prostate158_test':
        return sorted(os.listdir(ROOT_P158_TEST))
    raise ValueError(source)
