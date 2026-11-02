"""Stage 2: three-fold training and pooled held-out evaluation."""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BCRDataset, CLINICAL_TASKS
from model import build_model

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SPLIT_KFOLD = f'{REPO}/data/split_kfold.csv'
CKPT_BASE = f'{REPO}/checkpoints_stage2'
CLINICAL_LOSS = {'gleason_sum': nn.MSELoss, 'ece_epe': nn.BCEWithLogitsLoss, 'lni': nn.BCEWithLogitsLoss,
                 'biopsy_gg': nn.MSELoss, 'log_psa': nn.MSELoss}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed):
    return torch.Generator().manual_seed(seed)


def unpack(batch, use_tumor, use_clinical, device):
    tumor_mask, tumor_present, clinical = None, None, None
    if use_tumor or use_clinical:
        volumes, present, label, _, info = batch
        if use_tumor:
            tumor_mask = info['mask'].to(device, non_blocking=True)
            tumor_present = info['present'].to(device, non_blocking=True)
        if use_clinical:
            clinical = {'values': {t: v.to(device, non_blocking=True) for t, v in info['clinical_values'].items()},
                        'valid': {t: v.to(device, non_blocking=True) for t, v in info['clinical_valid'].items()}}
    else:
        volumes, present, label, _ = batch
    volumes = {m: v.to(device, non_blocking=True) for m, v in volumes.items()}
    present = {m: p.to(device, non_blocking=True) for m, p in present.items()}
    return volumes, present, label, tumor_mask, tumor_present, clinical


def clinical_aux_loss(aux_logits, clinical, criteria):
    total = torch.zeros((), device=next(iter(aux_logits.values())).device)
    for task, logit in aux_logits.items():
        valid = clinical['valid'][task]
        if valid.sum() == 0:
            continue
        per_sample = criteria[task](logit.float(), clinical['values'][task])
        total = total + (per_sample * valid).sum() / valid.sum()
    return total


def run_fold(fold, train_pids, test_pids, modalities, args, device):
    seed_everything(args.seed)
    use_lora = not (args.linear_probe or args.freeze_lora)
    use_tumor = args.tumor_pool_weight > 0
    use_clinical = args.aux_clinical
    overrides = None
    if args.no_ssl:
        overrides = {m: f'{REPO}/checkpoints_pretrained/Triad-SwinB-MAE.pth' for m in modalities}
    model = build_model(modalities=list(modalities), fusion_type=args.fusion_type, lora_rank=args.lora_rank,
                        lora_alpha=args.lora_alpha, use_lora=use_lora, linear_probe_mode=args.linear_probe,
                        tumor_pool_weight=args.tumor_pool_weight, aux_clinical=use_clinical, clinical_tasks=CLINICAL_TASKS,
                        ssl_ckpt_overrides=overrides, use_pos_encoding=not args.no_pos_encoding,
                        attn_mode=args.attn_mode).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    train_ds = BCRDataset(train_pids, modalities, augment=True, load_tumor_mask=use_tumor, load_clinical=use_clinical,
                          rotate_aug=args.rotate_aug, rotate_max_deg=args.rotate_max_deg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False,
                              generator=make_generator(args.seed))
    test_ds = BCRDataset(test_pids, modalities, load_tumor_mask=use_tumor, load_clinical=use_clinical)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False,
                             generator=make_generator(args.seed))
    train_labels = np.array([train_ds.labels[p] for p in train_pids])
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0 * n_neg / max(n_pos, 1)], device=device))
    clinical_criteria = {t: loss(reduction='none') for t, loss in CLINICAL_LOSS.items()}
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            volumes, present, label, tumor_mask, tumor_present, clinical = unpack(batch, use_tumor, use_clinical, device)
            label = label.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                if use_clinical:
                    logit, aux = model(volumes, present=present, tumor_mask=tumor_mask, tumor_present=tumor_present, return_aux=True)
                else:
                    logit = model(volumes, present=present, tumor_mask=tumor_mask, tumor_present=tumor_present)
                loss = criterion(logit, label)
                if use_clinical:
                    loss = loss + args.aux_clinical_weight * clinical_aux_loss(aux, clinical, clinical_criteria)
                if args.l2_weight > 0:
                    loss = loss + args.l2_weight * sum((p ** 2).sum() for p in trainable)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(label)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f'  fold {fold} epoch {epoch + 1}/{args.epochs}  train_loss={running / len(train_pids):.4f}')
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            pids = batch[3]
            volumes, present, label, tumor_mask, tumor_present, _ = unpack(batch, use_tumor, use_clinical, device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logit = model(volumes, present=present, tumor_mask=tumor_mask, tumor_present=tumor_present)
            prob = torch.sigmoid(logit).cpu().numpy()
            for pid, p, y in zip(pids, prob, label.numpy()):
                rows.append({'patient_id': pid, 'label': int(y), 'prob': float(p), 'fold': fold})
    return rows, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--modalities', default='t2,adc,pet', choices=['t2,adc,pet', 't2,adc', 't2,pet', 'adc,pet', 't2', 'adc', 'pet'])
    p.add_argument('--fusion_type', default='cross_attention', choices=['cross_attention', 'concat_mlp'])
    p.add_argument('--attn_mode', default='cross', choices=['cross', 'joint'])
    p.add_argument('--tumor_pool_weight', type=float, default=0.0)
    p.add_argument('--no_pos_encoding', action='store_true')
    p.add_argument('--no_ssl', action='store_true', help='use the Triad weights instead of the SimMIM checkpoints')
    p.add_argument('--freeze_lora', action='store_true', help='frozen encoders, as in the proposed model')
    p.add_argument('--linear_probe', action='store_true')
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--aux_clinical', action='store_true')
    p.add_argument('--aux_clinical_weight', type=float, default=0.3)
    p.add_argument('--rotate_aug', action='store_true')
    p.add_argument('--rotate_max_deg', type=float, default=3.0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--l2_weight', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2016)
    p.add_argument('--save_checkpoints', action='store_true')
    args = p.parse_args()
    modalities = tuple(args.modalities.split(','))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_df = pd.read_csv(SPLIT_KFOLD, dtype={'patient_id': str})
    ckpt_dir = f'{CKPT_BASE}/{args.tag}'
    if args.save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)
    all_rows = []
    for fold in sorted(fold_df['fold'].unique()):
        test_pids = fold_df[fold_df['fold'] == fold]['patient_id'].tolist()
        train_pids = fold_df[fold_df['fold'] != fold]['patient_id'].tolist()
        print(f'--- fold {fold}: train={len(train_pids)} test={len(test_pids)} ---')
        rows, model = run_fold(fold, train_pids, test_pids, modalities, args, device)
        all_rows.extend(rows)
        if args.save_checkpoints:
            torch.save(model.state_dict(), f'{ckpt_dir}/fold{fold}.pth')
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(all_rows)
    out_path = f'{REPO}/data/predictions_{args.tag}.csv'
    df.to_csv(out_path, index=False)
    print(f"n={len(df)}  positive={int(df['label'].sum())}")
    print(f"ROC-AUC={roc_auc_score(df['label'], df['prob']):.4f}  PR-AUC={average_precision_score(df['label'], df['prob']):.4f}")
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
