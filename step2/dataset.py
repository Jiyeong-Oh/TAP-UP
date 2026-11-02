"""Dataset over the cached paired cohort."""
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate
from torch.utils.data import Dataset

from clinical_sheets import load_sheet_clinical

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MANIFEST = f'{REPO}/data/bcr_manifest.csv'
MODALITIES = ('t2', 'adc', 'pet')
CLINICAL_TASKS = ('gleason_sum', 'ece_epe', 'lni', 'biopsy_gg', 'log_psa')


def _rotate3d(arr, angle_deg, order):
    if angle_deg == 0:
        return arr
    return rotate(arr, angle_deg, axes=(0, 1), reshape=False, order=order, mode='constant', cval=0.0).astype(np.float32)


class BCRDataset(Dataset):

    def __init__(self, pids, modalities=None, augment=False, load_tumor_mask=False, load_clinical=False,
                 rotate_aug=False, rotate_max_deg=3.0):
        manifest = pd.read_csv(MANIFEST, dtype={'patient_id': str})
        self.pids = list(pids)
        self.modalities = modalities if modalities is not None else MODALITIES
        self.augment = augment
        self.load_tumor_mask = load_tumor_mask
        self.load_clinical = load_clinical
        self.rotate_aug = rotate_aug
        self.rotate_max_deg = rotate_max_deg
        self.paths, self.labels = {}, {}
        for pid in self.pids:
            rows = manifest[manifest['patient_id'] == pid]
            self.paths[pid] = dict(zip(rows['modality'], rows['npy_path']))
            self.labels[pid] = int(rows.iloc[0]['bcr_label_binary'])
        if load_clinical:
            tab = load_sheet_clinical()
            self.clinical = {}
            for pid in self.pids:
                row = tab.loc[pid] if pid in tab.index else None
                raw = {t: row[t] if row is not None else np.nan for t in ('gleason_sum', 'ece_epe', 'lni')}
                raw['biopsy_gg'] = row['isup_gg'] - 3.0 if row is not None else np.nan
                raw['log_psa'] = (np.log(row['psa']) - 2.0) / 0.5 if row is not None and row['psa'] > 0 else np.nan
                self.clinical[pid] = {'values': {t: float(v) if pd.notna(v) else 0.0 for t, v in raw.items()},
                                      'valid': {t: float(pd.notna(v)) for t, v in raw.items()}}

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        do_rotate = self.augment and self.rotate_aug and self.rotate_max_deg > 0
        angle = float(np.random.uniform(-self.rotate_max_deg, self.rotate_max_deg)) if do_rotate else 0.0
        volumes, present = {}, {}
        for m in self.modalities:
            path = self.paths[pid].get(m)
            if path is None:
                volumes[m] = torch.zeros(1, 128, 128, 64, dtype=torch.float32)
                present[m] = torch.tensor(False)
                continue
            volumes[m] = torch.from_numpy(_rotate3d(np.load(path).astype(np.float32), angle, order=1)[None])
            present[m] = torch.tensor(True)
        label = torch.tensor(self.labels[pid], dtype=torch.float32)
        if not (self.load_tumor_mask or self.load_clinical):
            return volumes, present, label, pid
        tumor_path = self.paths[pid].get('tumor')
        if tumor_path is None:
            tumor_mask = torch.zeros(1, 128, 128, 64, dtype=torch.float32)
            tumor_present = torch.tensor(False)
        else:
            arr = _rotate3d(np.load(tumor_path).astype(np.float32), angle, order=0)
            tumor_mask = torch.from_numpy(arr).unsqueeze(0)
            tumor_present = torch.tensor(arr.sum() > 0)
        info = {'mask': tumor_mask, 'present': tumor_present}
        if self.load_clinical:
            c = self.clinical[pid]
            info['clinical_values'] = {t: torch.tensor(c['values'][t], dtype=torch.float32) for t in CLINICAL_TASKS}
            info['clinical_valid'] = {t: torch.tensor(c['valid'][t], dtype=torch.float32) for t in CLINICAL_TASKS}
        return volumes, present, label, pid, info
