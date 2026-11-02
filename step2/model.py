"""Frozen modality encoders, tumor-anchored fusion and the BCR classifier."""
import math
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f'{REPO}/external_repos/Triad')
from simmim_advanced import SwinTransformerSkipForSimMIM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora import inject_lora_swin, count_trainable

SSL_CKPT_DIR = f'{REPO}/checkpoints_ssl'
EMBED_DIM = 48
FEAT_DIM = EMBED_DIM * 16
HIDDEN_DIM, OUT_DIM, DROPOUT, NUM_HEADS = 128, 64, 0.4, 8


def compute_pool_weight(batch_size, d, h, w, device, dtype, tumor_pool_weight=0.0, tumor_mask=None, tumor_present=None):
    """Eq. 2: w_i = lambda t_i / sum_j t_j + (1 - lambda) / N; uniform without a tumor mask."""
    n = d * h * w
    weight = torch.full((batch_size, 1, n), 1.0 / n, device=device, dtype=dtype)
    if tumor_pool_weight > 0 and tumor_mask is not None:
        tumor_small = nn.functional.adaptive_avg_pool3d(tumor_mask, (d, h, w)).flatten(2)
        tumor_norm = tumor_small / (tumor_small.sum(dim=2, keepdim=True) + 1e-08)
        blended = tumor_pool_weight * tumor_norm + (1 - tumor_pool_weight) * weight
        if tumor_present is not None:
            flag = tumor_present.to(weight.dtype).view(-1, 1, 1)
            blended = blended * flag + weight * (1 - flag)
        weight = blended
    return weight


class TriadBranch(nn.Module):
    """One modality encoder initialized from its pretrained checkpoint."""

    def __init__(self, modality, lora_rank=16, lora_alpha=32, use_lora=False, tumor_pool_weight=0.0, ssl_ckpt_override=None):
        super().__init__()
        self.modality = modality
        self.ssl_ckpt_override = ssl_ckpt_override
        self.swin = SwinTransformerSkipForSimMIM(embed_dim=EMBED_DIM, in_chans=1)
        self._load_ssl_checkpoint()
        if use_lora:
            inject_lora_swin(self.swin, rank=lora_rank, alpha=lora_alpha)
        else:
            for p in self.swin.parameters():
                p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.tumor_pool_weight = tumor_pool_weight

    def _load_ssl_checkpoint(self):
        path = self.ssl_ckpt_override or f'{SSL_CKPT_DIR}/triad_{self.modality}_ssl.pth'
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt.get('model_state', ckpt)) if isinstance(ckpt, dict) else ckpt
        prefix = 'encoder.' if any(k.startswith('encoder.') for k in sd) else 'backbone.'
        msg = self.swin.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, strict=False)
        print(f'{self.modality} branch: loaded {path} (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})')

    def forward(self, x, return_features=False, tumor_mask=None, tumor_present=None):
        x4_out, _ = self.swin(x, mask=None, choice='none')
        if self.tumor_pool_weight > 0 and tumor_mask is not None:
            b, _, d, h, w = x4_out.shape
            weight = compute_pool_weight(b, d, h, w, x4_out.device, x4_out.dtype, self.tumor_pool_weight, tumor_mask, tumor_present)
            pooled = (x4_out.flatten(2) * weight).sum(dim=2)
        else:
            pooled = self.pool(x4_out).flatten(1)
        return (pooled, x4_out) if return_features else pooled


def _mlp(in_dim):
    return nn.Sequential(nn.Linear(in_dim, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU(), nn.Dropout(DROPOUT),
                         nn.Linear(HIDDEN_DIM, OUT_DIM), nn.LayerNorm(OUT_DIM), nn.GELU())


class ConcatMLPFusion(nn.Module):
    """Comparison: concatenation of the pooled features."""

    def __init__(self, n_branches):
        super().__init__()
        self.mlp = _mlp(n_branches * FEAT_DIM)

    def forward(self, feats):
        return self.mlp(torch.cat(feats, dim=1))


def sincos_pos_encoding(n, dim, device):
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CrossAttentionFusion(nn.Module):
    """Eqs. 1-2: shared positional encoding, cross-modal attention, tumor-anchored pooling.

    attn_mode='joint' lets a branch also attend to itself; single-modality models need it.
    """

    def __init__(self, n_branches, tumor_pool_weight=0.0, use_pos_encoding=True, attn_mode='cross'):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.attn_mode = attn_mode
        self.n_branches = n_branches
        self.tumor_pool_weight = tumor_pool_weight
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(FEAT_DIM, NUM_HEADS, dropout=DROPOUT, batch_first=True)
                                         for _ in range(n_branches)])
        self.norm = nn.ModuleList([nn.LayerNorm(FEAT_DIM) for _ in range(n_branches)])
        self.mlp = _mlp(n_branches * FEAT_DIM)

    def forward(self, token_seqs, token_grid_shape=None, tumor_mask=None, tumor_present=None):
        n, dim = token_seqs[0].shape[1], token_seqs[0].shape[2]
        if self.use_pos_encoding:
            pe = sincos_pos_encoding(n, dim, token_seqs[0].device).unsqueeze(0)
            token_seqs = [t + pe for t in token_seqs]
        use_weighted = token_grid_shape is not None and self.tumor_pool_weight > 0 and tumor_mask is not None
        if use_weighted:
            d, h, w = token_grid_shape
            weight_t = compute_pool_weight(token_seqs[0].shape[0], d, h, w, token_seqs[0].device, token_seqs[0].dtype,
                                           self.tumor_pool_weight, tumor_mask, tumor_present).transpose(1, 2)
        pooled = []
        for i, q in enumerate(token_seqs):
            kv = torch.cat([token_seqs[j] for j in range(self.n_branches) if j != i or self.attn_mode == 'joint'], dim=1)
            attn_out, _ = self.cross_attn[i](q, kv, kv)
            refined = self.norm[i](q + attn_out)
            pooled.append((refined * weight_t).sum(dim=1) if use_weighted else refined.mean(dim=1))
        return self.mlp(torch.cat(pooled, dim=1))


class BCRClassifier(nn.Module):

    def __init__(self, modalities=None, fusion_type='cross_attention', lora_rank=16, lora_alpha=32, use_lora=False,
                 linear_probe_mode=False, tumor_pool_weight=0.0, aux_clinical=False, clinical_tasks=None,
                 ssl_ckpt_overrides=None, use_pos_encoding=True, attn_mode='cross'):
        super().__init__()
        self.modalities = list(modalities) if modalities is not None else ['t2', 'adc', 'pet']
        self.fusion_type = fusion_type
        self.linear_probe_mode = linear_probe_mode
        overrides = ssl_ckpt_overrides or {}
        self.branches = nn.ModuleDict({m: TriadBranch(m, lora_rank, lora_alpha, use_lora, tumor_pool_weight, overrides.get(m))
                                       for m in self.modalities})
        self.missing_embed = nn.ParameterDict({m: nn.Parameter(torch.zeros(FEAT_DIM)) for m in self.modalities})
        n = len(self.modalities)
        if linear_probe_mode:
            self.fusion = nn.Identity()
            self.head = nn.Linear(n * FEAT_DIM, 1)
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(n, tumor_pool_weight, use_pos_encoding, attn_mode)
            self.head = nn.Linear(OUT_DIM, 1)
        else:
            self.fusion = ConcatMLPFusion(n)
            self.head = nn.Linear(OUT_DIM, 1)
        if aux_clinical:
            self.clinical_heads = nn.ModuleDict({t: nn.Linear(self.head.in_features, 1) for t in clinical_tasks})

    def _token_grid(self, x4_out):
        return x4_out.shape[2:]

    def _branch_tokens(self, m, x):
        _, x4_out = self.branches[m](x, return_features=True)
        return x4_out.flatten(2).transpose(1, 2), x4_out

    def forward(self, volumes, present=None, tumor_mask=None, tumor_present=None, return_aux=False):
        if self.fusion_type == 'cross_attention':
            token_seqs, grid = [], None
            for m in self.modalities:
                tokens, x4_out = self._branch_tokens(m, volumes[m])
                grid = self._token_grid(x4_out)
                if present is not None:
                    mask = present[m].to(tokens.dtype).view(-1, 1, 1)
                    tokens = tokens * mask + self.missing_embed[m].view(1, 1, -1) * (1 - mask)
                token_seqs.append(tokens)
            fused = self.fusion(token_seqs, grid, tumor_mask=tumor_mask, tumor_present=tumor_present)
        else:
            feats = []
            for m in self.modalities:
                feat = self.branches[m](volumes[m], tumor_mask=tumor_mask, tumor_present=tumor_present)
                if present is not None:
                    mask = present[m].to(feat.dtype).unsqueeze(1)
                    feat = feat * mask + self.missing_embed[m].unsqueeze(0) * (1 - mask)
                feats.append(feat)
            fused = torch.cat(feats, dim=1) if self.linear_probe_mode else self.fusion(feats)
        logit = self.head(fused).squeeze(-1)
        if return_aux:
            return logit, {t: head(fused).squeeze(-1) for t, head in self.clinical_heads.items()}
        return logit


def build_model(**kwargs):
    model = BCRClassifier(**kwargs)
    for m in model.modalities:
        trainable, total = count_trainable(model.branches[m])
        print(f'{m} branch: {trainable:,} / {total:,} trainable ({100 * trainable / total:.2f}%)')
    return model
