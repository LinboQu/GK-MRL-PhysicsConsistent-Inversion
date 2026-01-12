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
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, n: int = 1):
        self.sum += float(v) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)