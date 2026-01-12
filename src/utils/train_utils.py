from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def alpha_stats(a: torch.Tensor):
    # a: [B,K,T,H,W] or [B,K,*]
    eps = 1e-12
    # mean over B,T,H,W
    dims = tuple(i for i in range(a.ndim) if i != 1)  # keep K
    mean_k = a.mean(dim=dims)
    max_k  = a.amax(dim=dims)
    ent = -(a * (a.clamp_min(eps).log())).sum(dim=1)  # [B,T,H,W]
    ent_mean = ent.mean().item()
    return mean_k.detach().cpu().numpy(), max_k.detach().cpu().numpy(), ent_mean


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        v = float(val)
        self.val = v
        self.sum += v * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)