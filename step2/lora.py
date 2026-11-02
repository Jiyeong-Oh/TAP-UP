"""LoRA injection for the Swin attention layers (comparison method)."""
import os
import torch
import torch.nn as nn



class LoRALinear(nn.Module):


    def __init__(self, base: nn.Linear, rank: int=16, alpha: int=32, dropout: float=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(base.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.drop(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out



def inject_lora_swin(swin_module, rank=16, alpha=32, targets=('qkv', 'proj'), dropout=0.0):
    for p in swin_module.parameters():
        p.requires_grad = False
    n_wrapped = 0
    for (name, module) in swin_module.named_modules():
        if type(module).__name__ != 'WindowAttention':
            continue
        for target in targets:
            base = getattr(module, target, None)
            if base is None or isinstance(base, LoRALinear):
                continue
            setattr(module, target, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
            n_wrapped += 1
    print(f'inject_lora_swin: wrapped {n_wrapped} linear layers across all WindowAttention blocks (rank={rank}, alpha={alpha}, targets={targets})')
    return swin_module



def count_trainable(module):
    trainable = sum((p.numel() for p in module.parameters() if p.requires_grad))
    total = sum((p.numel() for p in module.parameters()))
    return (trainable, total)
