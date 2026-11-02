"""Clinical comparison model (logistic regression)."""
import os
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VARS = ['age', 'psa', 'psa_density', 'isup_gg']
GLEASON_TO_GG = {'3+3': 1, '3+4': 2, '4+3': 3, '4+4 OR 3+5': 4, '4+4': 4, '3+5': 4, '5+3': 4, '>=4+5': 5, '4+5': 5, '5+4': 5, '5+5': 5}



def yesno(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    return 1.0 if s in ('yes', '1', '1.0') else 0.0 if s in ('no', '0', '0.0') else np.nan



def load_clinical():
    c = pd.read_csv(f'{REPO}/data/clinical.csv', dtype={'patient_id': str})
    c['psa_density'] = c['psa'] / c['volume']
    c = c[['patient_id', 'age', 'psa', 'volume', 'isup_gg', 'bcr_sheet', 'psa_density']]
    return c.drop_duplicates('patient_id')



def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    (pos, neg) = (np.where(y == 1)[0], np.where(y == 0)[0])
    a = [roc_auc_score(y[b], s[b]) for b in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n))]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5))



def paired_delta(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    (pos, neg) = (np.where(y == 1)[0], np.where(y == 0)[0])
    d = []
    for _ in range(n):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[b], s1[b]) - roc_auc_score(y[b], s2[b]))
    return (roc_auc_score(y, s1) - roc_auc_score(y, s2), np.percentile(d, 2.5), np.percentile(d, 97.5))



def youden(y, s):
    (fpr, tpr, thr) = roc_curve(y, s)
    pred = s >= thr[np.argmax(tpr - fpr)]
    return (int((pred & (y == 1)).sum()), int((y == 1).sum()), int((~pred & (y == 0)).sum()), int((y == 0).sum()))



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--imaging_tag', default='tapup')
    p.add_argument('--no_log', action='store_true')
    p.add_argument('--out_csv', default=f'{REPO}/data/clinical_baseline_preop_oof.csv')
    args = p.parse_args()
    folds = pd.read_csv(f'{REPO}/data/split_kfold.csv')
    df = folds.merge(load_clinical(), on='patient_id', how='left')
    y = df['bcr_label_binary'].values.astype(int)
    print(f"n={len(df)}  BCR+={y.sum()}  matched in the sheets: {df['age'].notna().sum() + 0}/{len(df)}")
    print('missing per variable:', df[VARS].isna().sum().to_dict())
    miss = df[df[VARS].isna().any(axis=1)]
    if len(miss):
        print('patients with a missing variable:', ', '.join((f"{r.patient_id}(BCR{('+' if r.bcr_label_binary else '-')})" for r in miss.itertuples())))
    agree = df.dropna(subset=['bcr_sheet'])
    dis = agree[agree['bcr_sheet'] != agree['bcr_label_binary']]
    print(f'BCR label in the sheet vs. our label: {len(agree) - len(dis)}/{len(agree)} agree' + (f"; differ: {', '.join((f'{r.patient_id} (ours {r.bcr_label_binary}, sheet {int(r.bcr_sheet)})' for r in dis.itertuples()))}" if len(dis) else ''))
    print('\nmedian (IQR) by BCR:')
    for v in VARS:
        q = df.groupby('bcr_label_binary')[v].quantile([0.25, 0.5, 0.75]).unstack()
        print(f'  {v:12s} BCR-: {q.loc[0, 0.5]:.2f} ({q.loc[0, 0.25]:.2f}-{q.loc[0, 0.75]:.2f})   BCR+: {q.loc[1, 0.5]:.2f} ({q.loc[1, 0.25]:.2f}-{q.loc[1, 0.75]:.2f})')
    print('\nSingle variable as score (no fitting; missing dropped):')
    for v in VARS:
        ok = df[v].notna().values
        print(f'  {v:12s} n={ok.sum()}  AUC={roc_auc_score(y[ok], df[v].values[ok]):.3f}')
    X = df[VARS].copy()
    if not args.no_log:
        X['psa'] = np.log(X['psa'])
        X['psa_density'] = np.log(X['psa_density'])
    oof = np.zeros(len(df))
    for f in sorted(df['fold'].unique()):
        (tr, te) = ((df['fold'] != f).values, (df['fold'] == f).values)
        med = X[tr].median()
        (xtr, xte) = (X[tr].fillna(med).values, X[te].fillna(med).values)
        (mu, sd) = (xtr.mean(0), xtr.std(0) + 1e-08)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit((xtr - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((xte - mu) / sd)[:, 1]
        print(f'  fold {f}: coefficients (standardised) ' + ', '.join((f'{v}={c:+.2f}' for (v, c) in zip(VARS, clf.coef_[0]))))
    img = pd.read_csv(f'{REPO}/data/predictions_{args.imaging_tag}.csv').set_index('patient_id')
    s_img = img.loc[df['patient_id'], 'prob'].values
    rows = []
    for (name, s) in (('Imaging model (T2W + ADC + PSMA-PET)', s_img), (f"Clinical model (age + {('log ' if not args.no_log else '')}PSA + {('log ' if not args.no_log else '')}PSA density + biopsy ISUP GG), logistic regression", oof)):
        (lo, hi) = boot_ci(y, s)
        (tp, npos, tn, nneg) = youden(y, s)
        row = {'model': name, 'n': len(y), 'auc': roc_auc_score(y, s), 'auc_lo': lo, 'auc_hi': hi, 'sens': f'{tp / npos:.3f} ({tp}/{npos})', 'spec': f'{tn / nneg:.3f} ({tn}/{nneg})'}
        if s is oof:
            (d, dlo, dhi) = paired_delta(y, oof, s_img)
            row.update({'delta_vs_imaging': d, 'delta_lo': dlo, 'delta_hi': dhi})
        rows.append(row)
    print()
    for r in rows:
        dtxt = f"  dAUC vs imaging {r['delta_vs_imaging']:+.3f} ({r['delta_lo']:+.3f}, {r['delta_hi']:+.3f})" if 'delta_vs_imaging' in r else ''
        print(f"{r['model']}\n  n={r['n']}  AUC {r['auc']:.3f} ({r['auc_lo']:.3f}-{r['auc_hi']:.3f})  sens {r['sens']}  spec {r['spec']}{dtxt}")
    print(f'\ncorrelation clinical vs imaging OOF scores: r={np.corrcoef(oof, s_img)[0, 1]:.3f}')
    df[['patient_id', 'fold']].assign(label=y, clinical_prob=oof, imaging_prob=s_img, **{v: df[v] for v in VARS}).to_csv(args.out_csv, index=False)
    print(f'wrote {args.out_csv}')
if __name__ == '__main__':
    main()
