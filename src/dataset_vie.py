from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.facies_io import read_facies_intervals_txt, intervals_to_facies_series
import torch
from torch.utils.data import Dataset

import segyio

from .geo_constraints import DataPaths, read_segy_cube


# -------------------------
# Utils
# -------------------------
def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_u = [c.upper() for c in cols]
    for cand in candidates:
        if cand.upper() in cols_u:
            return cols[cols_u.index(cand.upper())]
    return None


def read_wellheads(wellheads_csv: str) -> pd.DataFrame:
    """
    Read wellhead table and normalize key columns.

    Expected to contain at least:
      - WELLNAME
      - either (INLINE,XLINE) or (X,Y)
    """
    df = pd.read_csv(wellheads_csv)
    df.columns = [c.strip().upper() for c in df.columns]

    name_col = _pick_col(df.columns.tolist(), ["WELLNAME", "WELL", "NAME"])
    if name_col is None:
        raise ValueError(f"Cannot find well name column in {wellheads_csv}. Columns={df.columns.tolist()}")

    il_col = _pick_col(df.columns.tolist(), ["INLINE", "IL"])
    xl_col = _pick_col(df.columns.tolist(), ["XLINE", "XL"])

    x_col = _pick_col(df.columns.tolist(), ["X", "EASTING"])
    y_col = _pick_col(df.columns.tolist(), ["Y", "NORTHING"])

    df = df.rename(columns={name_col: "WELLNAME"})
    if il_col: df = df.rename(columns={il_col: "INLINE"})
    if xl_col: df = df.rename(columns={xl_col: "XLINE"})
    if x_col:  df = df.rename(columns={x_col: "X"})
    if y_col:  df = df.rename(columns={y_col: "Y"})

    if ("INLINE" not in df.columns or "XLINE" not in df.columns) and ("X" not in df.columns or "Y" not in df.columns):
        raise ValueError(
            "wellheads must contain either (INLINE,XLINE) or (X,Y). "
            f"Columns={df.columns.tolist()}"
        )
    return df


def ilxl_to_index(il: int, xl: int) -> Tuple[int, int]:
    """
    Your cube uses IL=1..150, XL=1..200. Convert to 0-based indices.
    """
    return il - 1, xl - 1


def extract_25d_patch(vol: np.ndarray, il0: int, xl0: int, hw: int) -> np.ndarray:
    """
    vol: [C,IL,XL,T] or [IL,XL,T]
    return patch:
      - if vol is [IL,XL,T] -> [1, 2hw+1, 2hw+1, T]
      - if vol is [C,IL,XL,T] -> [C, 2hw+1, 2hw+1, T]
    Pads by edge values if near boundary.
    """
    if vol.ndim == 3:
        vol = vol[None, ...]  # [1,IL,XL,T]
    C, n_il, n_xl, n_t = vol.shape

    i0, i1 = il0 - hw, il0 + hw + 1
    j0, j1 = xl0 - hw, xl0 + hw + 1

    # pad if needed
    pad_i0 = max(0, -i0)
    pad_j0 = max(0, -j0)
    pad_i1 = max(0, i1 - n_il)
    pad_j1 = max(0, j1 - n_xl)

    i0c, i1c = max(i0, 0), min(i1, n_il)
    j0c, j1c = max(j0, 0), min(j1, n_xl)

    patch = vol[:, i0c:i1c, j0c:j1c, :]  # [C,*,*,T]

    if pad_i0 or pad_i1 or pad_j0 or pad_j1:
        patch = np.pad(
            patch,
            pad_width=((0, 0), (pad_i0, pad_i1), (pad_j0, pad_j1), (0, 0)),
            mode="edge"
        )

    return patch.astype(np.float32)


def load_constraints_npz(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# -------------------------
# Sample index builder
# -------------------------
@dataclass
class SampleIndex:
    wellname: str
    il: int   # 1-based
    xl: int   # 1-based

def build_well_sample_index(wellheads_csv: str) -> List[SampleIndex]:
    df = read_wellheads(wellheads_csv)
    # prefer INLINE/XLINE
    if "INLINE" in df.columns and "XLINE" in df.columns:
        out = []
        for _, r in df.iterrows():
            out.append(SampleIndex(str(r["WELLNAME"]), int(r["INLINE"]), int(r["XLINE"])))
        return out
    else:
        raise ValueError("This dataset builder currently requires INLINE/XLINE in wellheads.csv for exact cube indexing.")


# -------------------------
# PyTorch Dataset
# -------------------------
class StanfordVIEWellPatchDataset(Dataset):
    """
    Returns a dict:
      - x: seismic patch         [1, H, W, T]
      - p: prior tensor patch    [4, H, W, T]
      - c: reliability patch     [1, H, W, T]
      - m: mask patch            [1, H, W, T]
      - y: AI center trace       [T]  (supervised impedance label for the center (il,xl))
    """
    def __init__(
        self,
        paths: DataPaths,
        constraints_npz: str,
        patch_hw: int = 4,
        use_masked_y: bool = True,
        normalize: bool = True,
    ):
        self.paths = paths
        self.patch_hw = int(patch_hw)
        self.use_masked_y = bool(use_masked_y)
        self.normalize = bool(normalize)

        # Load volumes
        self.seis = read_segy_cube(paths.segy_seis)  # [IL,XL,T]
        self.ai   = read_segy_cube(paths.segy_ai)    # [IL,XL,T]

        # Load constraints
        pack = load_constraints_npz(constraints_npz)
        self.P = pack["P"]               # [4,IL,XL,T]
        self.C = pack["C"]               # [IL,XL,T]
        self.M = pack["M"]               # [IL,XL,T]

        # Build sample index from wellheads
        self.samples = build_well_sample_index(paths.wellheads_csv)

        # load facies series for each well (cached)
        self.facies_map = {}
        t_ms = np.arange(self.seis.shape[-1], dtype=np.float32)  # 0..199
        for s in self.samples:
            fac_path = os.path.join(self.paths.virtual_facies_dir, f"{s.wellname}.txt")
            if os.path.exists(fac_path):
                intervals = read_facies_intervals_txt(fac_path)
                fac_series = intervals_to_facies_series(intervals, t_ms=t_ms, default_label=-1)  # [T]
            else:
                fac_series = np.full((self.seis.shape[-1],), -1, dtype=np.int64)
            self.facies_map[s.wellname] = fac_series

        # Simple normalization stats (global) – robust & reproducible
        if self.normalize:
            self.seis_mean = float(np.mean(self.seis))
            self.seis_std  = float(np.std(self.seis) + 1e-8)
            self.ai_mean   = float(np.mean(self.ai))
            self.ai_std    = float(np.std(self.ai) + 1e-8)
        else:
            self.seis_mean = self.seis_std = 0.0
            self.ai_mean = self.ai_std = 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        il0, xl0 = ilxl_to_index(s.il, s.xl)

        # patches
        x_patch = extract_25d_patch(self.seis, il0, xl0, self.patch_hw)         # [1,H,W,T]
        p_patch = extract_25d_patch(self.P,    il0, xl0, self.patch_hw)         # [4,H,W,T]
        c_patch = extract_25d_patch(self.C,    il0, xl0, self.patch_hw)         # [1,H,W,T]
        m_patch = extract_25d_patch(self.M,    il0, xl0, self.patch_hw)         # [1,H,W,T]
        # center reliability for weighting losses
        H = c_patch.shape[1]
        W = c_patch.shape[2]
        c_center = c_patch[0, H//2, W//2, :]  # [T]

        facies = self.facies_map.get(s.wellname, None)
        if facies is None:
            facies = np.full((self.seis.shape[-1],), -1, dtype=np.int64)

        # valid mask: facies label exists AND within interval mask
        m_center = self.M[il0, xl0, :].astype(np.float32)  # [T]
        facies_valid = ((facies >= 0).astype(np.float32) * m_center).astype(np.float32)  # [T]

        # label: center AI trace
        y = self.ai[il0, xl0, :].astype(np.float32)  # [T]
        if self.use_masked_y:
            # mask outside target interval
            m_center = self.M[il0, xl0, :].astype(np.float32)
            y = y * m_center

        # normalize
        if self.normalize:
            x_patch = (x_patch - self.seis_mean) / self.seis_std
            y = (y - self.ai_mean) / self.ai_std

        # torch: keep [C,H,W,T] for now; later you can permute to [C,T,H,W] if your net wants
        out = {
            "x": torch.from_numpy(x_patch),  # [1,H,W,T]
            "p": torch.from_numpy(p_patch),  # [4,H,W,T]
            "c": torch.from_numpy(c_patch),  # [1,H,W,T]
            "m": torch.from_numpy(m_patch),  # [1,H,W,T]
            "y": torch.from_numpy(y),        # [T]
            "c_center": torch.from_numpy(c_center.astype(np.float32)),  # [T]
            "wellname": s.wellname,
            "il": torch.tensor(s.il, dtype=torch.int32),
            "xl": torch.tensor(s.xl, dtype=torch.int32),
            "facies_center": torch.from_numpy(facies.astype(np.int64)),        # [T]  values in {0,1,2,3} or -1
            "facies_valid":  torch.from_numpy(facies_valid.astype(np.float32)) # [T]  0/1
        }
        return out