from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# SSAM: Stratigraphic Scale Alignment Module
# Align multi-scale features {E_{s-1}, E_s, E_{s+1}} to the same (T,H,W) grid of scale s.
# Paper-like: (Up/Down) + Conv(1×k×k) + Conv(1×1×1) + BN + ReLU
# -------------------------
class SSAM(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.conv_k = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=(1, k, k), padding=(0, k//2, k//2), bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv3d(cout, cout, kernel_size=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, size_t: int, size_h: int, size_w: int) -> torch.Tensor:
        """
        x: [B,C,T,H,W] at some scale
        output aligned: [B,C_out,T,size_h,size_w]
        We keep T aligned too (interpolate), but you can choose to keep T unchanged across encoder.
        """
        # align (T,H,W)
        x = F.interpolate(x, size=(size_t, size_h, size_w), mode="trilinear", align_corners=False)
        x = self.conv_k(x)
        x = self.conv_1(x)
        return x


# -------------------------
# GSSM: Geology-guided Scale Selection Module
# Paper-like: hierarchical attention + softmax weights W_e -> alpha^(s)
# We implement voxel-wise softmax over scales (element-wise product + addition).
# -------------------------
class GSSM(nn.Module):
    def __init__(self, f_ch: int, p_ch: int, hidden: int = 32):
        super().__init__()
        # score network produces per-voxel logits for each scale branch
        self.score = nn.Sequential(
            nn.Conv3d(f_ch + p_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, feats: List[torch.Tensor], p_s: torch.Tensor):
        """
        return:
            out:   [B,F,T,H,W]
            alpha: [B,S,T,H,W]  (S = number of branches)
        """
        logits = []
        for f in feats:
            z = torch.cat([f, p_s], dim=1)
            logits.append(self.score(z))
        logits = torch.cat(logits, dim=1)      # [B,S,T,H,W]
        alpha = torch.softmax(logits, dim=1)   # [B,S,T,H,W]

        out = 0.0
        for i, f in enumerate(feats):
            out = out + f * alpha[:, i:i+1, ...]
        return out, alpha



# -------------------------
# PSFM block at scale s
# Inputs: multi-scale encoder features (e_{s-1}, e_s, e_{s+1})
# Prior:  p (full prior tensor) -> downsample+projection -> p^(s)
# Output: F_mod^(s)
# -------------------------
class PSFM(nn.Module):
    def __init__(
        self,
        ch_s: int,                # channel at scale s after alignment
        ch_prev: Optional[int],   # channel at scale s-1
        ch_next: Optional[int],   # channel at scale s+1
        p_in: int = 4,            # input prior channels (your P has 4)
        p_embed: int = 16,        # projection dim for P^(s)
        k: int = 3,
    ):
        super().__init__()
        self.has_prev = ch_prev is not None
        self.has_next = ch_next is not None

        # SSAM aligners (to ch_s)
        if self.has_prev:
            self.ssam_prev = SSAM(ch_prev, ch_s, k=k)
        else:
            self.ssam_prev = None

        self.ssam_self = SSAM(ch_s, ch_s, k=k)

        if self.has_next:
            self.ssam_next = SSAM(ch_next, ch_s, k=k)
        else:
            self.ssam_next = None

        # P^(s): downsample + projection
        self.p_proj = nn.Sequential(
            nn.Conv3d(p_in, p_embed, kernel_size=1, bias=False),
            nn.BatchNorm3d(p_embed),
            nn.ReLU(inplace=True),
        )

        # GSSM chooses scales under geology guidance
        self.gssm = GSSM(f_ch=ch_s, p_ch=p_embed)

    def _project_prior(self, p: torch.Tensor, size: torch.Size) -> torch.Tensor:
        """
        p: [B, P_in, H, W, T]  (your dataset format)
        size: (T,H,W) in BCTHW conv space
        return p_s: [B, P_embed, T,H,W]
        """
        # to [B,P,T,H,W]
        p = p.permute(0, 1, 4, 2, 3).contiguous()
        p = F.interpolate(p, size=size, mode="trilinear", align_corners=False)
        p = self.p_proj(p)
        return p

    def forward(
        self,
        e_prev: Optional[torch.Tensor],  # [B,Cprev,T,H,W] or None
        e_self: torch.Tensor,            # [B,Cs,T,H,W]
        e_next: Optional[torch.Tensor],  # [B,Cnext,T,H,W] or None
        p_full: torch.Tensor             # [B,Pin,H,W,T]  (same as model input p)
    ) -> torch.Tensor:
        B, Cs, T, H, W = e_self.shape
        # P^(s)
        p_s = self._project_prior(p_full, size=(T, H, W))  # [B,Pembed,T,H,W]

        # SSAM align all to (T,H,W) and Cs channels
        feats = []

        if self.has_prev and e_prev is not None:
            feats.append(self.ssam_prev(e_prev, T, H, W))
        feats.append(self.ssam_self(e_self, T, H, W))
        if self.has_next and e_next is not None:
            feats.append(self.ssam_next(e_next, T, H, W))

        # GSSM fusion with α^(s)
        f_mod, alpha = self.gssm(feats, p_s)
        
        # cache for debugging/analysis (do not backprop through this cache)
        self._last_alpha = alpha.detach()
        
        return f_mod
