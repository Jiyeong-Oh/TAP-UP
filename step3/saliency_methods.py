"""Integrated gradients with a fold-mean baseline."""
import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

def _forward_logit(model, vols, tumor_mask, tumor_present):
    (seqs, grid) = ([], None)
    b = next(iter(vols.values())).shape[0]
    for m in model.modalities:
        (tokens, x4) = model._branch_tokens(m, vols[m])
        grid = model._token_grid(x4)
        seqs.append(tokens)
    fused = model.fusion(seqs, grid, tumor_mask=tumor_mask.expand(b, -1, -1, -1, -1), tumor_present=tumor_present.expand(b))
    return model.head(fused).squeeze(-1)

def _norm(a):
    a = np.clip(a, 0, None)
    return a / (a.max() + 1e-12)

def blurred(vol, sigma=4.0):
    return torch.from_numpy(gaussian_filter(vol[0, 0].detach().cpu().numpy(), sigma=sigma)[None, None]).to(vol)

def fold_mean_volumes(train_pids, device):
    from dataset import BCRDataset, MODALITIES
    ds = BCRDataset(pids=list(train_pids), modalities=MODALITIES, load_tumor_mask=False, augment=False)
    acc = None
    for j in range(len(ds)):
        v = ds[j][0]
        acc = {m: v[m].clone() for m in MODALITIES} if acc is None else {m: acc[m] + v[m] for m in MODALITIES}
    return {m: (t / len(ds)).unsqueeze(0).to(device) for (m, t) in acc.items()}

def integrated_gradients(model, vols, tumor_mask, tumor_present, steps=32, chunk=4, sigma=4.0, baseline=None):
    model.eval()
    base = baseline if baseline is not None else {m: blurred(v, sigma) for (m, v) in vols.items()}
    grads = {m: torch.zeros_like(v) for (m, v) in vols.items()}
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    for s in range(0, steps, chunk):
        a = alphas[s:s + chunk].to(next(iter(vols.values())).device).view(-1, 1, 1, 1, 1)
        xs = {m: (base[m] + a * (vols[m] - base[m])).requires_grad_(True) for m in vols}
        logit = _forward_logit(model, xs, tumor_mask, tumor_present)
        g = torch.autograd.grad(logit.sum(), [xs[m] for m in model.modalities])
        for (m, gm) in zip(model.modalities, g):
            grads[m] += gm.detach().sum(dim=0, keepdim=True)
    attr = {m: ((vols[m] - base[m]) * grads[m] / steps)[0, 0].detach().cpu().numpy() for m in model.modalities}
    return ({m: _norm(a) for (m, a) in attr.items()}, attr)

def compute(method, model, vols, tumor_mask, tumor_present, present=None, baseline=None, raw=False):
    if method not in ('ig', 'ig_mean'):
        raise ValueError(method)
    if method == 'ig_mean' and baseline is None:
        raise ValueError('ig_mean needs baseline=fold_mean_volumes(...)')
    out = integrated_gradients(model, vols, tumor_mask, tumor_present, baseline=baseline if method == 'ig_mean' else None)
    return out[1] if raw else out[0]
