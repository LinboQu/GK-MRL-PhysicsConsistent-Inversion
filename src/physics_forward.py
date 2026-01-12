from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def ricker_wavelet(dt_ms: float, f0_hz: float, nt: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Ricker wavelet centered at t=0.
    dt_ms: sampling interval in ms
    f0_hz: dominant frequency in Hz
    nt: number of samples
    return: [nt]
    """
    dt = dt_ms / 1000.0  # s
    t0 = (nt - 1) / 2.0 * dt
    t = torch.arange(nt, device=device, dtype=dtype) * dt - t0
    pi2 = (math.pi ** 2)
    a = (math.pi * f0_hz) ** 2
    w = (1.0 - 2.0 * a * t**2) * torch.exp(-a * t**2)
    return w


def ai_to_reflectivity(ai: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    ai: [B,T] normalized or physical AI (both ok, but MUST be positive-ish if physical)
    Use stable log-derivative reflectivity surrogate.
    return r: [B,T] with r[0]=0 (padding)
    """
    # shift to avoid log instability; for normalized AI, values can be negative -> we use softplus
    # This keeps differentiability and avoids taking log of negatives.
    ai_pos = F.softplus(ai) + eps
    log_ai = torch.log(ai_pos)
    # first difference along time
    d = log_ai[:, 1:] - log_ai[:, :-1]  # [B,T-1]
    r = F.pad(d, (1, 0), mode="constant", value=0.0)  # [B,T]
    return r


def convolve1d_same(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T]
    w: [K] wavelet
    return y: [B,T] with 'same' length using conv1d.
    """
    B, T = x.shape
    K = w.numel()
    # conv1d expects [B,C,T]
    x1 = x[:, None, :]  # [B,1,T]
    w1 = w.flip(0)[None, None, :]  # [1,1,K] flip for conv convention
    pad = K // 2
    y = F.conv1d(x1, w1, padding=pad)  # [B,1,T]
    return y[:, 0, :]


def forward_seismic_from_ai(
    ai: torch.Tensor,
    dt_ms: float = 1.0,
    f0_hz: float = 40.0,
    wavelet_nt: int = 81,
) -> torch.Tensor:
    """
    ai: [B,T] predicted AI (normalized ok)
    return s_syn: [B,T] synthetic seismic
    """
    r = ai_to_reflectivity(ai)  # [B,T]
    w = ricker_wavelet(dt_ms=dt_ms, f0_hz=f0_hz, nt=wavelet_nt, device=ai.device, dtype=ai.dtype)  # [K]
    s = convolve1d_same(r, w)   # [B,T]
    return s