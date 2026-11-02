"""Gland-box crop, isotropic resampling and resizing."""
import os
import numpy as np
from scipy.ndimage import zoom, distance_transform_edt



def resample_to_isotropic(volume, current_spacing, target_spacing=1.0, order=1):
    factors = [cs / target_spacing for cs in current_spacing]
    return zoom(volume, factors, order=order)



def crop_around_mask(volume, mask, margin_mm, spacing=(1.0, 1.0, 1.0)):
    if mask.sum() == 0:
        return None
    dilated = distance_transform_edt(mask < 0.5, sampling=spacing) <= margin_mm
    coords = np.array(np.where(dilated))
    lo = coords.min(axis=1)
    hi = coords.max(axis=1) + 1
    sl = tuple((slice(int(l), int(h)) for (l, h) in zip(lo, hi)))
    return (volume[sl], mask[sl], (lo.tolist(), hi.tolist()))



def resize_to_shape(volume, target_shape, order=1):
    factors = [t / s for (t, s) in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)
