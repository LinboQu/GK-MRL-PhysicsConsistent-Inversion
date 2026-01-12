from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoCNNTraceHead(nn.Module):
    """
    Baseline: 3D CNN over (T,H,W) using 2.5D spatial patch + time axis.
    Input:  x [B,1,H,W,T]
            p [B,4,H,W,T]
            c [B,1,H,W,T]
            m [B,1,H,W,T]
    Output: y_hat [B,T]  (center trace prediction)
    """
    def __init__(self, in_channels: int = 7, base: int = 32, t: int = 200):
        super().__init__()
        self.t = t

        def block(cin, cout, k=(3,3,3), s=(1,1,1), p=(1,1,1)):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        # encoder
        self.enc1 = nn.Sequential(
            block(in_channels, base),
            block(base, base),
        )
        self.down1 = block(base, base*2, s=(2,2,2))      # down T,H,W
        self.enc2 = nn.Sequential(
            block(base*2, base*2),
            block(base*2, base*2),
        )
        self.down2 = block(base*2, base*4, s=(2,2,2))
        self.enc3 = nn.Sequential(
            block(base*4, base*4),
            block(base*4, base*4),
        )

        # global pooling to trace
        self.pool = nn.AdaptiveAvgPool3d((t, 1, 1))  # keep T, pool H,W to 1
        self.head = nn.Sequential(
            nn.Conv3d(base*4, base*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base*2, 1, kernel_size=1),
        )

    def forward(self, x, p, c, m):
        # x/p/c/m: [B,C,H,W,T] -> [B,C,T,H,W]
        def to_bcthw(z):
            return z.permute(0, 1, 4, 2, 3).contiguous()

        x = to_bcthw(x)
        p = to_bcthw(p)
        c = to_bcthw(c)
        m = to_bcthw(m)

        inp = torch.cat([x, p, c, m], dim=1)  # [B,7,T,H,W]

        f = self.enc1(inp)
        f = self.down1(f)
        f = self.enc2(f)
        f = self.down2(f)
        f = self.enc3(f)

        f = self.pool(f)      # [B,C,T,1,1]
        y = self.head(f)      # [B,1,T,1,1]
        y = y[:, 0, :, 0, 0]  # [B,T]
        return y