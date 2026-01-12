from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.psfm import PSFM


class GeoCNNMultiTask(nn.Module):
    """
    Encoder with PSFM (SSAM+GSSM) at multiple scales.
    Input:  x [B,1,H,W,T], p [B,4,H,W,T], c [B,1,H,W,T], m [B,1,H,W,T]
    Output: ai_hat [B,T], facies_logits [B,K,T]
    """
    def __init__(self, in_channels: int = 7, base: int = 32, t: int = 200, n_facies: int = 4):
        super().__init__()
        self.t = t
        self.n_facies = n_facies
        self._last_alphas = None

        def conv3(cin, cout, k=(3,3,3), s=(1,1,1), p=(1,1,1)):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        # ---------- encoder backbone (3 scales, stable for 9x9 patch) ----------
        self.enc1 = nn.Sequential(conv3(in_channels, base), conv3(base, base))
        self.down1 = conv3(base, base*2, s=(2,2,2))  # down T,H,W
        self.enc2 = nn.Sequential(conv3(base*2, base*2), conv3(base*2, base*2))
        self.down2 = conv3(base*2, base*4, s=(2,2,2))
        self.enc3 = nn.Sequential(conv3(base*4, base*4), conv3(base*4, base*4))

        # ---------- PSFM at each scale s ----------
        # scale1: no prev, has next
        self.psfm1 = PSFM(ch_s=base,   ch_prev=None,     ch_next=base*2, p_in=4, p_embed=16, k=3)
        # scale2: has prev and next
        self.psfm2 = PSFM(ch_s=base*2, ch_prev=base,     ch_next=base*4, p_in=4, p_embed=16, k=3)
        # scale3: has prev, no next
        self.psfm3 = PSFM(ch_s=base*4, ch_prev=base*2,   ch_next=None,   p_in=4, p_embed=16, k=3)

        # ---------- heads ----------
        # pool H,W to 1, keep T
        self.pool = nn.AdaptiveAvgPool3d((t, 1, 1))
        self.ai_head = nn.Sequential(
            nn.Conv3d(base*4, base*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base*2, 1, kernel_size=1),
        )
        self.facies_head = nn.Sequential(
            nn.Conv3d(base*4, base*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base*2, n_facies, kernel_size=1),
        )

    @staticmethod
    def _to_bcthw(z: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W,T] -> [B,C,T,H,W]
        return z.permute(0, 1, 4, 2, 3).contiguous()

    def get_last_psfm_alphas(self):
        """
        Return cached alphas from psfm1/2/3 for debugging.
        Each alpha is [B,S,T,H,W] where S is number of branches.
        """
        return {
            "psfm1": getattr(self.psfm1, "_last_alpha", None),
            "psfm2": getattr(self.psfm2, "_last_alpha", None),
            "psfm3": getattr(self.psfm3, "_last_alpha", None),
        }

    def get_alphas(self):
        return self._last_alphas

    def forward(self, x, p, c, m):
        x = self._to_bcthw(x)
        p_bcthw = self._to_bcthw(p)   # only needed for shapes? PSFM expects original p format, we pass original
        c = self._to_bcthw(c)
        m = self._to_bcthw(m)

        # NOTE: PSFM takes p in original [B,4,H,W,T] format (paper P^(s) from P)
        # so keep the original p tensor too
        p_full = p  # [B,4,H,W,T]

        inp = torch.cat([x, self._to_bcthw(p_full), c, m], dim=1)  # [B,7,T,H,W]

        e1 = self.enc1(inp)                 # [B,base,T,H,W]
        d1 = self.down1(e1)                 # [B,2b,T/2,H/2,W/2]
        e2 = self.enc2(d1)                  # [B,2b,...]
        d2 = self.down2(e2)                 # [B,4b,...]
        e3 = self.enc3(d2)                  # [B,4b,...]

        # PSFM requires multi-scale features aligned to each scale
        # For scale1: use e1 (self) and e2 (next); align e2 up to e1 resolution
        e1m = self.psfm1(e_prev=None, e_self=e1, e_next=e2, p_full=p_full)
        # For scale2: use e1 (prev) and e2 (self) and e3 (next)
        e2m = self.psfm2(e_prev=e1,  e_self=e2, e_next=e3, p_full=p_full)
        # For scale3: use e2 (prev) and e3 (self)
        e3m = self.psfm3(e_prev=e2,  e_self=e3, e_next=None, p_full=p_full)

        self._last_alphas = {
            "psfm1": getattr(self.psfm1, "_last_alpha", None),
            "psfm2": getattr(self.psfm2, "_last_alpha", None),
            "psfm3": getattr(self.psfm3, "_last_alpha", None),
        }

        # We continue with the deepest geology-modulated feature (paper: local features feed decoder)
        f = e3m
        f = self.pool(f)  # [B,4b,T,1,1]

        ai = self.ai_head(f)          # [B,1,T,1,1]
        fac = self.facies_head(f)     # [B,K,T,1,1]

        ai = ai[:, 0, :, 0, 0]        # [B,T]
        fac = fac[:, :, :, 0, 0]      # [B,K,T]
        return ai, fac
