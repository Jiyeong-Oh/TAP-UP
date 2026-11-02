"""Comparison architectures trained on the same split."""
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import BCRDataset
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MODS = ('t2', 'adc', 'pet')
SWIN_SSL = f'{REPO}/checkpoints_pretrained/model_swinvit.pt'
MASS_DIR = f'{REPO}/external_repos/MASS'
MASS_CKPT = f'{REPO}/checkpoints_pretrained/mass_base.pth'
MASS_CFG = dict(in_ch=1, channels=[32, 64, 128, 256, 512], scale=[2, 2, 2, 2], kernel_size=[3, 3, 3, 3, 3], block='BasicBlock', num_block=[2, 2, 2, 2], pool=True, norm='in', tn=73, num_prior_stage=3, ema_moment=0.99)



class AttentionPooling(nn.Module):


    def __init__(self, c):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv3d(c, c // 4, 1), nn.ReLU(inplace=True), nn.Conv3d(c // 4, 1, 1))


    def forward(self, x):
        a = torch.softmax(self.attention(x).view(x.size(0), -1), dim=1).view(x.size(0), 1, *x.shape[2:])
        return (x * a).sum(dim=(2, 3, 4))



class TumorPooling(nn.Module):


    def __init__(self, w):
        super().__init__()
        self.w = w


    def forward(self, x, tmask, tpres):
        t = F.adaptive_avg_pool3d(tmask.float(), x.shape[2:]).flatten(1)
        n = t.size(1)
        t = t / t.sum(1, keepdim=True).clamp_min(1e-08)
        a = self.w * t + (1 - self.w) / n
        a = torch.where(tpres.view(-1, 1).bool(), a, torch.full_like(a, 1 / n))
        return (x.flatten(2) * a.unsqueeze(1)).sum(2)



class MassClassifier(nn.Module):


    def __init__(self, n_mod=3, tumor_pool_weight=0.0):
        super().__init__()
        import sys
        if MASS_DIR not in sys.path:
            sys.path.insert(0, MASS_DIR)
        from models.iris import Iris
        full = Iris(**MASS_CFG)
        ck = torch.load(MASS_CKPT, map_location='cpu')
        sd = ck.get('ema_model', ck.get('model', ck.get('model_state_dict', ck)))
        sd = {k.replace('module.', ''): v for (k, v) in sd.items()}
        own = full.encoder.state_dict()
        enc = {k[len('encoder.'):]: v for (k, v) in sd.items() if k.startswith('encoder.')}
        missing = [k for k in own if k not in enc or enc[k].shape != own[k].shape]
        if missing:
            raise RuntimeError(f'{len(missing)} MASS encoder tensors not loaded: {missing[:5]}')
        full.encoder.load_state_dict(enc)
        print(f"MASS encoder: {len(enc)}/{len(own)} tensors loaded from {MASS_CKPT} ({('ema_model' if 'ema_model' in ck else 'model')})")
        self.encoder = full.encoder.eval()
        for q in self.encoder.parameters():
            q.requires_grad = False
        self.tumor_pool = TumorPooling(tumor_pool_weight) if tumor_pool_weight > 0 else None
        self.heads = nn.ModuleList([nn.Sequential(nn.Identity() if self.tumor_pool else AttentionPooling(512), nn.Dropout(0.1), nn.Linear(512, 1)) for _ in range(n_mod)])


    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


    def forward(self, x, tmask=None, tpres=None):
        out = 0
        for (k, head) in enumerate(self.heads):
            with torch.no_grad():
                f = self.encoder(x[:, k:k + 1])[0]
            f = f.float()
            if self.tumor_pool:
                f = self.tumor_pool(f, tmask, tpres)
            out = out + head(f).squeeze(-1)
        return out



class SwinClassifier(nn.Module):


    def __init__(self, in_ch=3, ssl_path=SWIN_SSL):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        net = SwinUNETR(in_channels=in_ch, out_channels=1, feature_size=48, spatial_dims=3)
        if ssl_path and os.path.exists(ssl_path):
            sd = torch.load(ssl_path, map_location='cpu')
            sd = sd.get('state_dict', sd)
            sd = {k.replace('module.', '').replace('.mlp.fc1.', '.mlp.linear1.').replace('.mlp.fc2.', '.mlp.linear2.'): v for (k, v) in sd.items()}
            own = net.swinViT.state_dict()
            w = 'patch_embed.proj.weight'
            if w in sd and sd[w].shape[1] != own[w].shape[1]:
                sd[w] = sd[w].repeat(1, own[w].shape[1], 1, 1, 1) / own[w].shape[1]
            hits = {k: v for (k, v) in sd.items() if k in own and own[k].shape == v.shape}
            net.swinViT.load_state_dict(hits, strict=False)
            print(f'Swin UNETR SSL weights: {len(hits)}/{len(own)} tensors loaded from {ssl_path}')
            missing = [k for k in own if k not in hits and (not k.endswith('relative_position_index'))]
            if missing:
                raise RuntimeError(f'{len(missing)} encoder tensors not initialised from the SSL checkpoint: {missing[:5]}')
        else:
            raise FileNotFoundError(f'Swin UNETR SSL weights not found: {ssl_path}')
        self.encoder = net.swinViT
        self.head = nn.Linear(16 * 48, 1)


    def forward(self, x):
        h = self.encoder(x, normalize=True)[-1]
        return self.head(h.mean(dim=(2, 3, 4))).squeeze(-1)



def build(name, tumor_pool_weight=0.0):
    if name == 'resnet18':
        from monai.networks.nets import resnet18
        net = resnet18(spatial_dims=3, n_input_channels=3, num_classes=1)
        return nn.Sequential(net, nn.Flatten(0))
    if name == 'densenet121':
        from monai.networks.nets import DenseNet121
        return nn.Sequential(DenseNet121(spatial_dims=3, in_channels=3, out_channels=1), nn.Flatten(0))
    if name == 'swinunetr':
        return SwinClassifier()
    if name == 'mass':
        return MassClassifier(tumor_pool_weight=tumor_pool_weight)
    raise ValueError(name)



def stack(volumes):
    return torch.cat([volumes[m] for m in MODS], dim=1)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True, choices=['resnet18', 'densenet121', 'swinunetr', 'mass'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--rotate_max_deg', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=2016)
    p.add_argument('--tag', default=None)
    args = p.parse_args()
    assert args.tumor_pool_weight == 0 or args.backbone == 'mass'
    use_tumor = args.tumor_pool_weight > 0
    fwd = lambda model, x, batch: model(x, batch[4]['mask'].to(device), batch[4]['present'].to(device)) if use_tumor else model(x)
    device = torch.device('cuda')
    split = pd.read_csv(f'{REPO}/data/split_kfold.csv')
    rows = []
    for fold in sorted(split['fold'].unique()):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tr = split[split['fold'] != fold]['patient_id'].tolist()
        te = split[split['fold'] == fold]['patient_id'].tolist()
        g = torch.Generator()
        g.manual_seed(args.seed)
        tr_dl = DataLoader(BCRDataset(pids=tr, modalities=MODS, augment=True, rotate_aug=True, rotate_max_deg=args.rotate_max_deg, load_tumor_mask=use_tumor), batch_size=args.batch_size, shuffle=True, num_workers=2, generator=g)
        te_ds = BCRDataset(pids=te, modalities=MODS, augment=False, load_tumor_mask=use_tumor)
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        y_tr = np.array([split.set_index('patient_id').loc[p_, 'bcr_label_binary'] for p_ in tr])
        model = build(args.backbone, args.tumor_pool_weight).to(device)
        n_par = sum((q.numel() for q in model.parameters() if q.requires_grad))
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=device))
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler()
        print(f'fold {fold}: {args.backbone}, {n_par / 1000000.0:.1f}M trainable params, train {len(tr)} test {len(te)}', flush=True)
        for ep in range(args.epochs):
            model.train()
            tot = 0.0
            for batch in tr_dl:
                (volumes, label) = (batch[0], batch[2])
                (x, yb) = (stack(volumes).to(device), label.to(device))
                opt.zero_grad()
                with torch.autocast(device_type='cuda'):
                    loss = crit(fwd(model, x, batch).float(), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(yb)
            if (ep + 1) % 5 == 0:
                print(f'  fold {fold} epoch {ep + 1}/{args.epochs} train_loss={tot / len(tr):.4f}', flush=True)
        model.eval()
        with torch.no_grad():
            for batch in te_dl:
                (volumes, label, pids) = (batch[0], batch[2], batch[3])
                with torch.autocast(device_type='cuda'):
                    prob = torch.sigmoid(fwd(model, stack(volumes).to(device), batch).float()).cpu().numpy()
                rows += [{'patient_id': q, 'label': int(l), 'prob': float(pr), 'fold': int(fold)} for (q, l, pr) in zip(pids, label.numpy(), prob)]
        del model
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    path = f'{REPO}/data/predictions_baseline_{args.tag or args.backbone}.csv'
    out.to_csv(path, index=False)
    from sklearn.metrics import roc_auc_score
    print(f'pooled OOF AUC={roc_auc_score(out.label, out.prob):.4f}  -> {path}')
if __name__ == '__main__':
    main()
