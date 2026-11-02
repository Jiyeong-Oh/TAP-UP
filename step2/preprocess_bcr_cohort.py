"""Preprocess the paired cohort and cache it as .npy with a manifest."""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'step1'))
import data_sources as ds
from preprocess_cohort import process_patient
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CACHE_DIR = os.environ.get('TAPUP_CACHE', 'cache') + '/paired'
MANIFEST_OUT = f'{REPO}/data/bcr_manifest.csv'



def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    import preprocess_cohort as pc
    pc.CACHE_DIR = CACHE_DIR
    labels = pd.read_csv(f'{REPO}/data/cohort.csv', dtype={'patient_id': str, 'cohort': str}).set_index('patient_id')
    rows = []
    for pid in labels.index:
        source = labels.loc[pid, 'cohort']
        try:
            pt_rows = []
            process_patient(source, pid, ds.load_internal(pid), pt_rows)
            for r in pt_rows:
                r['bcr_label_binary'] = int(labels.loc[pid, 'bcr_label_binary'])
                r['cohort'] = source
            rows.extend(pt_rows)
        except Exception as e:
            print(f'{source}/{pid}: FAILED {repr(e)}')
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_OUT, index=False)
    print(f'wrote {MANIFEST_OUT}: {len(df)} rows')
    print(df.groupby(['cohort', 'modality']).size())
if __name__ == '__main__':
    main()
