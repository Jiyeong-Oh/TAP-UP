"""Re-implementation of the Papp et al. radiomics model."""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f'{REPO}/step1')
FEATS = f'{REPO}/data/papp_radiomics_features.csv'
SEED = 2016
EXCLUDE = set()
BIN = {'pet': 0.05, 't2': 0.05, 'adc': 5.0}
STAGE = {'1': 1.0, '1a': 1.1, '1b': 1.2, '1c': 1.3, '2': 2.0, '2a': 2.1, '2b': 2.2, '2c': 2.3, '3': 3.0, '3a': 3.1, '3b': 3.2, '4': 4.0}
RF_TABLE = [(99, 10, 2, 1.0, False), (99, 10, None, 1.0, False), (99, 10, None, 0.9, False), (1000, 5, None, 1.0, True), (1000, 5, 3, 1.0, True), (1000, 10, None, 1.0, True), (1000, 5, None, 0.9, True), (1000, 5, 3, 0.9, True), (1000, 10, None, 0.9, True)]



def extract():
    import SimpleITK as sitk
    from radiomics import featureextractor
    from scipy.ndimage import uniform_filter, zoom
    import data_sources as ds
    split = pd.read_csv(f'{REPO}/data/split_kfold.csv')
    rows = []
    for pid in split['patient_id']:
        d = ds.load_internal(pid)
        gland = d['gland'] > 0.5
        tumor = d['tumor'] > 0.5 if d['tumor'] is not None else np.zeros_like(gland)
        voi_src = 'tumor' if tumor.sum() > 0 else 'gland'
        voi = tumor if voi_src == 'tumor' else gland
        ref = gland & ~voi if voi_src == 'tumor' else gland
        pet_ok = d['pet'][ref].mean() > 0
        vols = {'t2': d['t2'] / d['t2'][ref].mean(), 'adc': d['adc']}
        if pet_ok:
            vols['pet'] = d['pet'] / d['pet'][ref].mean()
        sp = tuple((float(s) for s in d['spacing']))

        def img(a):
            im = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)).astype(np.float32))
            im.SetSpacing(sp)
            return im
        mask = img(voi.astype(np.uint8))
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        row = {'patient_id': pid, 'voi': voi_src, 'n_vox_2mm': np.nan}
        for (m, a) in vols.items():
            ex = featureextractor.RadiomicsFeatureExtractor(binWidth=BIN[m], resampledPixelSpacing=[2, 2, 2], interpolator='sitkLinear', correctMask=True, label=1, minimumROIDimensions=1)
            ex.disableAllFeatures()
            ex.enableFeatureClassByName('firstorder')
            for c in ('glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm'):
                ex.enableFeatureClassByName(c)
            if m == 't2':
                ex.enableFeatureClassByName('shape')
            out = ex.execute(img(a), mask)
            for (k, v) in out.items():
                if k.startswith('original_'):
                    row[f"{m}_{k[len('original_'):]}"] = float(v)
            if m == 't2':
                row['n_vox_2mm'] = int(out['diagnostics_Mask-interpolated_VoxelNum'])
        if pet_ok:
            p = vols['pet']
            size = [max(1, int(round(12.4 / s))) for s in sp]
            pk = uniform_filter(p, size=size)
            vml = voi.sum() * np.prod(sp) / 1000.0
            row.update({'suv_max': p[voi].max(), 'suv_mean': p[voi].mean(), 'suv_peak': pk[voi].max(), 'suv_tlg': p[voi].mean() * vml})
        rows.append(row)
        print(pid, voi_src, row['n_vox_2mm'], 'features', len(row) - 3, flush=True)
    pd.DataFrame(rows).to_csv(FEATS, index=False)
    print('wrote', FEATS)



def load_targets():
    from clinical_baseline_preop import load_clinical
    c = pd.read_csv(f'{REPO}/data/clinical.csv', dtype={'patient_id': str})
    c = c[['patient_id', 'g1', 'g2', 'stage']].drop_duplicates('patient_id').set_index('patient_id')
    c['high_risk'] = ((c['g1'] >= 4) | (c['g1'] + c['g2'] >= 8)).astype(float).where(c[['g1', 'g2']].notna().all(axis=1))
    c['stage_enum'] = c['stage'].map(lambda v: STAGE.get(str(v).strip().lower().lstrip('ct'), np.nan) if pd.notna(v) else np.nan)
    psa = load_clinical().set_index('patient_id')['psa']
    c['psa'] = psa.reindex(c.index)
    return c[['high_risk', 'stage_enum', 'psa']]



def fit_rf(X, y, trees, leaf, mf, bag, oob, seed):
    rf = RandomForestClassifier(n_estimators=trees, criterion='entropy', max_depth=5, min_samples_leaf=leaf, max_features=mf, bootstrap=True, max_samples=None if bag == 1.0 else bag, random_state=seed, n_jobs=-1).fit(X, y)
    if oob:
        acc = []
        for (est, idx) in zip(rf.estimators_, rf.estimators_samples_):
            o = np.setdiff1d(np.arange(len(y)), idx)
            acc.append((est.predict(X[o]) == rf.classes_.searchsorted(y[o])).mean() if len(o) else 0.0)
        keep = np.argsort(acc)[::-1][:99]
        rf.estimators_ = [rf.estimators_[i] for i in keep]
        rf.n_estimators = 99
    return rf



def ensemble_score(Xtr, ytr, Xte, seed=SEED):
    return np.mean([fit_rf(Xtr, ytr, *cfg, seed=seed + k).predict_proba(Xte)[:, 1] for (k, cfg) in enumerate(RF_TABLE)], axis=0)



def select(Xtr, ytr, names, k=5):
    ok = np.nanstd(Xtr, axis=0) > 0
    (Xtr, names) = (Xtr[:, ok], np.array(names)[ok])
    r2 = np.array([np.corrcoef(Xtr[:, j], ytr)[0, 1] ** 2 for j in range(Xtr.shape[1])])
    r2 = np.nan_to_num(r2)
    (order, C, kept) = (np.argsort(-r2), np.abs(np.corrcoef(Xtr, rowvar=False)), [])
    for j in order:
        if all((C[j, i] <= 0.75 for i in kept)):
            kept.append(j)
    return (list(names[kept[:k]]), len(kept))



def oof_lesion_scores(F, y, tr, te, inner_seed=SEED):
    names = [c for c in F.columns if c not in ('voi', 'n_vox_2mm')]
    F = F.copy()
    F[names] = F[names].fillna(F.loc[tr, names].median())
    lab = tr[~np.isnan(y.loc[tr].values)]
    (sel, n_red) = select(F.loc[lab, names].values, y.loc[lab].values, names)
    te_score = ensemble_score(F.loc[lab, sel].values, y.loc[lab].values, F.loc[te, sel].values)
    tr_score = pd.Series(np.nan, index=tr)
    skf = StratifiedKFold(5, shuffle=True, random_state=inner_seed)
    for (a, b) in skf.split(lab, y.loc[lab].values):
        (s2, _) = select(F.loc[lab[a], names].values, y.loc[lab[a]].values, names)
        tr_score.loc[lab[b]] = ensemble_score(F.loc[lab[a], s2].values, y.loc[lab[a]].values, F.loc[lab[b], s2].values)
    rest = tr_score.index[tr_score.isna()]
    if len(rest):
        tr_score.loc[rest] = ensemble_score(F.loc[lab, sel].values, y.loc[lab].values, F.loc[rest, sel].values)
    return (tr_score, pd.Series(te_score, index=te), sel, n_red)



def smote(X, y, seed=SEED):
    from imblearn.over_sampling import SMOTE
    k = int(min(5, (y == 1).sum() - 1))
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)



def youden(y, s):
    (fpr, tpr, thr) = roc_curve(y, s)
    j = np.argmax(tpr - fpr)
    return (tpr[j], 1 - fpr[j])



def model(keep_all=False):
    split = pd.read_csv(f'{REPO}/data/split_kfold.csv').set_index('patient_id')
    if not keep_all:
        split = split.drop(index=[p for p in EXCLUDE if p in split.index])
    print(f'{len(split)} patients ({int(split.bcr_label_binary.sum())} BCR), excluded {([] if keep_all else sorted(EXCLUDE))}')
    F = pd.read_csv(FEATS).set_index('patient_id').loc[split.index]
    T = load_targets().reindex(split.index)
    print('lesion label available', T['high_risk'].notna().sum(), 'high risk', int(T['high_risk'].sum()), '| stage missing', T['stage_enum'].isna().sum(), '| PSA missing', T['psa'].isna().sum(), '| VOI < 64 voxels at 2 mm:', int((F['n_vox_2mm'] < 64).sum()), '| gland VOI:', int((F['voi'] == 'gland').sum()))
    bcr = split['bcr_label_binary'].astype(int)
    out = {k: [] for k in ('papp2021', 'papp2021_clh', 'papp2021_direct')}
    names = [c for c in F.columns if c not in ('voi', 'n_vox_2mm')]
    for fold in sorted(split['fold'].unique()):
        tr = split.index[split['fold'] != fold].values
        te = split.index[split['fold'] == fold].values
        (clh_tr, clh_te, sel, n_red) = oof_lesion_scores(F, T['high_risk'], tr, te)
        print(f'fold {fold}: MLH features after redundancy reduction {n_red}, selected {sel}')
        C = T[['psa', 'stage_enum']].copy()
        med = C.loc[tr].median()
        C = C.fillna(med)
        Xtr = np.c_[C.loc[tr].values, clh_tr.loc[tr].values]
        Xte = np.c_[C.loc[te].values, clh_te.loc[te].values]
        (Xs, ys) = smote(Xtr, bcr.loc[tr].values)
        s_full = ensemble_score(Xs, ys, Xte)
        Fi = F[names].fillna(F.loc[tr, names].median())
        (dsel, _) = select(Fi.loc[tr, names].values, bcr.loc[tr].values, names)
        (Xs, ys) = smote(Fi.loc[tr, dsel].values, bcr.loc[tr].values)
        s_dir = ensemble_score(Xs, ys, Fi.loc[te, dsel].values)
        for (k, s) in (('papp2021', s_full), ('papp2021_clh', clh_te.loc[te].values), ('papp2021_direct', s_dir)):
            out[k] += [{'patient_id': p, 'label': int(bcr.loc[p]), 'prob': float(v), 'fold': int(fold)} for (p, v) in zip(te, s)]
    for (k, rows) in out.items():
        d = pd.DataFrame(rows)
        d.to_csv(f"{REPO}/data/predictions_baseline_{k}{('_84' if keep_all else '')}.csv", index=False)
        (se, sp) = youden(d.label, d.prob)
        print(f'{k:18s} AUC={roc_auc_score(d.label, d.prob):.3f}  SENS={se:.3f}  SPEC={sp:.3f}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['extract', 'model'])
    p.add_argument('--keep_all', action='store_true')
    a = p.parse_args()
    extract() if a.step == 'extract' else model(a.keep_all)
