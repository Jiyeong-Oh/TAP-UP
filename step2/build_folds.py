"""Stratified three-fold split."""
import os
import numpy as np
import pandas as pd
MANIFEST = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'data/cohort.csv')
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'data/split_kfold.csv')
N_FOLDS = 3
SEED = 2026



def main():
    df = pd.read_csv(MANIFEST)
    rng = np.random.RandomState(SEED)
    fold = {}
    for cohort in df['cohort'].unique():
        for label in (0, 1):
            pids = df[(df['cohort'] == cohort) & (df['bcr_label_binary'] == label)]['patient_id'].tolist()
            rng.shuffle(pids)
            for (i, pid) in enumerate(pids):
                fold[pid] = i % N_FOLDS + 1
    df['fold'] = df['patient_id'].map(fold)
    df[['patient_id', 'bcr_label_binary', 'cohort', 'fold']].to_csv(OUT, index=False)
    print(df.groupby(['fold', 'cohort', 'bcr_label_binary']).size())
    print()
    print(df.groupby('fold').size())
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
