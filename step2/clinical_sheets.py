"""Clinical targets used by the auxiliary heads."""
import os
import numpy as np
import pandas as pd
from clinical_baseline_preop import REPO, load_clinical



def _num(x):
    return pd.to_numeric(x, errors='coerce')



def _pad(s):
    return f'{s[:4]}-{int(s[5:]):02d}'



def load_pathology():
    c = pd.read_csv(f'{REPO}/data/clinical.csv', dtype={'patient_id': str})
    return c[['patient_id', 'gleason_sum', 'ece_epe', 'lni']].drop_duplicates('patient_id')



def load_sheet_clinical():
    return load_clinical().merge(load_pathology(), on='patient_id', how='outer').set_index('patient_id')
