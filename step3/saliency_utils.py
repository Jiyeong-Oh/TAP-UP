"""Shared helpers for the interpretation scripts."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f'{REPO}/step1')
sys.path.insert(0, f'{REPO}/step2')
import data_sources as ds
from crop_utils import crop_around_mask, resample_to_isotropic, resize_to_shape
from dataset import CLINICAL_TASKS
from model import build_model

MARGIN_MM, TARGET_SPACING, TARGET_SHAPE = 3.0, 0.5, (128, 128, 64)


def load_model(checkpoint, modalities, tumor_pool_weight=0.9, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(modalities=list(modalities), fusion_type='cross_attention', use_lora=False,
                        tumor_pool_weight=tumor_pool_weight, aux_clinical=True, clinical_tasks=CLINICAL_TASKS).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def gland_and_spacing(pid):
    """Gland mask on the model grid and the true voxel spacing of that grid in mm."""
    data = ds.load_internal(pid)
    gland, spacing = data['gland'], data['spacing']
    _, cropped, _ = crop_around_mask(gland, gland, margin_mm=MARGIN_MM, spacing=spacing)
    resampled = resample_to_isotropic(cropped, spacing, TARGET_SPACING, order=0)
    resized = resize_to_shape(resampled, TARGET_SHAPE, order=0)
    return resized, tuple(TARGET_SPACING * s / t for s, t in zip(resampled.shape, TARGET_SHAPE))


def outline(mask_2d, color=(0, 1, 0), thickness=1):
    m = mask_2d > 0.5
    rgba = np.zeros((*mask_2d.shape, 4))
    rgba[m & ~binary_erosion(m, iterations=thickness)] = [*color, 1.0]
    return rgba


def heatmap_rgba(sal_2d, cmap='hot', alpha_scale=0.85):
    rgba = plt.get_cmap(cmap)(sal_2d)
    rgba[..., 3] = sal_2d * alpha_scale
    return rgba
