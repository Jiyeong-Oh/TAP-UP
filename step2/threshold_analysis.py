"""Sensitivity and specificity at the Youden-optimal threshold."""
import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))



def analyze(tag):
    path = f'{REPO}/data/predictions_{tag}.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    (y, p) = (df['label'].values, df['prob'].values)
    auc = roc_auc_score(y, p)
    (fpr, tpr, thr) = roc_curve(y, p)
    spec = 1 - fpr
    j = tpr + spec - 1
    best = int(np.argmax(j))
    eer = int(np.argmin(np.abs(tpr - spec)))
    fixed_sens = ((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)
    fixed_spec = ((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return dict(tag=tag, n=len(y), n_pos=int(y.sum()), roc_auc=auc, sens_at_0_5=fixed_sens, spec_at_0_5=fixed_spec, youden_thr=float(thr[best]), youden_sens=float(tpr[best]), youden_spec=float(spec[best]), youden_j=float(j[best]), eer_thr=float(thr[eer]), eer_sens=float(tpr[eer]), eer_spec=float(spec[eer]))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='*')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        tags = sorted((os.path.basename(f)[len('predictions_'):-4] for f in glob.glob(f'{REPO}/data/predictions_*.csv')))
    else:
        tags = args.tags
    rows = [r for r in (analyze(t) for t in tags) if r is not None]
    if not rows:
        print('no matching prediction CSVs found')
        return
    out = pd.DataFrame(rows).sort_values('youden_j', ascending=False)
    pd.set_option('display.width', 200)
    print(out[['tag', 'roc_auc', 'sens_at_0_5', 'spec_at_0_5', 'youden_thr', 'youden_sens', 'youden_spec', 'youden_j', 'eer_thr', 'eer_sens', 'eer_spec']].to_string(index=False))
if __name__ == '__main__':
    main()
