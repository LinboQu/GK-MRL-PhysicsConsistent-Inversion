from __future__ import annotations

import numpy as np
import torch


def alpha_stats(a: torch.Tensor):
    # a: [B,K,T,H,W] or [B,K,*]
    eps = 1e-12
    # mean over B,T,H,W
    dims = tuple(i for i in range(a.ndim) if i != 1)  # keep K
    mean_k = a.mean(dim=dims)
    max_k = a.amax(dim=dims)
    ent = -(a * (a.clamp_min(eps).log())).sum(dim=1)  # [B,T,H,W]
    ent_mean = ent.mean().item()
    return mean_k.detach().cpu().numpy(), max_k.detach().cpu().numpy(), ent_mean
