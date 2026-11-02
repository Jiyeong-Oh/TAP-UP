"""Stage 1: modality-specific SimMIM pretraining of a Triad Swin-B encoder."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f'{REPO}/external_repos/Triad')
from simmim_advanced import SwinTransformerSkipForSimMIM, SimMIMSkip_light

MANIFEST = f'{REPO}/data/ssl_manifest.csv'
CKPT_IN = f'{REPO}/checkpoints_pretrained/Triad-SwinB-MAE.pth'
CKPT_OUT_DIR = f'{REPO}/checkpoints_ssl'
EMBED_DIM, FEATURE_SIZE = 48, 768
VOL_SHAPE = (128, 128, 64)


class MaskGenerator:

    def __init__(self, input_size=VOL_SHAPE, mask_patch_size=16, model_patch_size=2, mask_ratio=0.6):
        self.rand_size = tuple(s // mask_patch_size for s in input_size)
        self.scale = mask_patch_size // model_patch_size
        self.token_count = int(np.prod(self.rand_size))
        self.mask_count = int(np.ceil(self.token_count * mask_ratio))

    def __call__(self):
        mask = np.zeros(self.token_count, dtype=np.float32)
        mask[np.random.permutation(self.token_count)[:self.mask_count]] = 1
        mask = mask.reshape(self.rand_size)
        return mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)


class SSLVolumeDataset(Dataset):

    def __init__(self, modality, mask_ratio=0.6):
        df = pd.read_csv(MANIFEST)
        self.paths = df[df['modality'] == modality]['npy_path'].tolist()
        if len(self.paths) == 0:
            raise RuntimeError(f'No rows for modality={modality} in {MANIFEST}')
        self.mask_gen = MaskGenerator(mask_ratio=mask_ratio)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = np.load(self.paths[idx]).astype(np.float32)
        return torch.from_numpy(vol).unsqueeze(0), torch.from_numpy(self.mask_gen())


def build_model():
    encoder = SwinTransformerSkipForSimMIM(embed_dim=EMBED_DIM, in_chans=1, use_checkpoint=True)
    model = SimMIMSkip_light(in_channels=1, out_channels=1, encoder=encoder, feature_size=FEATURE_SIZE, choice='mae', ssl_weight=0.0)
    ckpt = torch.load(CKPT_IN, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt.get('model_state', ckpt)) if isinstance(ckpt, dict) else ckpt
    msg = model.load_state_dict({k.replace('backbone.', 'encoder.'): v for k, v in sd.items()}, strict=False)
    print(f'checkpoint load: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    return model


def cosine_lr(step, total_steps, base_lr, warmup_steps, min_lr=1e-06):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modality', required=True, choices=['t2', 'adc', 'pet'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--mask_ratio', type=float, default=0.6)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()
    os.makedirs(CKPT_OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SSLVolumeDataset(args.modality, mask_ratio=args.mask_ratio)
    print(f'modality={args.modality}: {len(dataset)} volumes')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)
    global_step = 0
    ckpt_path = f'{CKPT_OUT_DIR}/triad_{args.modality}_ssl.pth'
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for vol, mask in loader:
            lr = cosine_lr(global_step, total_steps, args.lr, warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr
            vol = vol.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = model(vol, mask, vol, None)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            global_step += 1
        avg_loss = running_loss / len(loader)
        print(f'[{args.modality}] epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.5f}  lr={lr:.2e}')
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}, ckpt_path)
    print(f'done. final checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
