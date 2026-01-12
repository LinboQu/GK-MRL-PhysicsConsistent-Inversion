from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import segyio


# -------------------------
# Paths (match your folder layout)
# -------------------------
@dataclass
class DataPaths:
    data_root: str

    @property
    def attr_time_sgy(self) -> str:
        return os.path.join(self.data_root, "ATTR_TIME_SGY")

    @property
    def seismic_horizon_dir(self) -> str:
        return os.path.join(self.data_root, "seismic_horizon")

    @property
    def virtual_wells_regular_dir(self) -> str:
        return os.path.join(self.data_root, "virtual_wells_regular")

    @property
    def virtual_wells_zonation_dir(self) -> str:
        return os.path.join(self.data_root, "virtual_wells_zonation")

    @property
    def virtual_facies_dir(self) -> str:
        # VW0001.txt ~ VW0300.txt live here
        return os.path.join(self.data_root, "virtual_facies_txt_noTop")

    # --- core SEG-Y paths ---
    @property
    def segy_seis(self) -> str:
        return os.path.join(
            self.attr_time_sgy,
            "synthetic_poststack_TIME_dt1ms_40Hz_fromAI_XY25.sgy"
        )

    @property
    def segy_ai(self) -> str:
        # FIXED: this should be Acoustic_Impedance_TIME_dt1ms_XY25.sgy
        return os.path.join(
            self.attr_time_sgy,
            "Acoustic_Impedance_TIME_dt1ms_XY25.sgy"
        )

    # --- wells / zonation ---
    @property
    def wellheads_csv(self) -> str:
        return os.path.join(self.virtual_wells_regular_dir, "virtual_wellheads.csv")

    @property
    def zonation_txt(self) -> str:
        return os.path.join(self.virtual_wells_zonation_dir, "zonation.txt")

    # --- facies ---
    @property
    def facies_dir(self) -> str:
        return self.virtual_facies_dir

    def facies_txt(self, wellname: str) -> str:
        # NOTE: cannot be @property because it takes an argument
        return os.path.join(self.facies_dir, f"{wellname}.txt")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_root, "processed")


# -------------------------
# SEG-Y IO
# -------------------------
@dataclass
class SeisGrid:
    ilines: np.ndarray
    xlines: np.ndarray
    twt_ms: np.ndarray
    dt_ms: float
    x2d: Optional[np.ndarray] = None
    y2d: Optional[np.ndarray] = None


def _try_read_xy_maps(
    f: segyio.SegyFile, n_il: int, n_xl: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    candidates = [
        (segyio.TraceField.SourceX, segyio.TraceField.SourceY),
        (segyio.TraceField.CDP_X, segyio.TraceField.CDP_Y),
        (segyio.TraceField.GroupX, segyio.TraceField.GroupY),
    ]
    for fx, fy in candidates:
        try:
            xs = np.array([f.header[i][fx] for i in range(f.tracecount)], dtype=np.float64)
            ys = np.array([f.header[i][fy] for i in range(f.tracecount)], dtype=np.float64)
            if np.all(xs == 0) and np.all(ys == 0):
                continue
            x2d = xs.reshape(n_il, n_xl)
            y2d = ys.reshape(n_il, n_xl)
            return x2d, y2d
        except Exception:
            continue
    return None, None


def read_segy_grid(segy_path: str) -> SeisGrid:
    """
    Read inline/xline/time axis from a 3D post-stack SEG-Y.
    Compatible with different segyio versions.
    """
    with segyio.open(segy_path, "r", ignore_geometry=False) as f:
        f.mmap()

        ilines = np.array(f.ilines, dtype=int)
        xlines = np.array(f.xlines, dtype=int)

        # robust time axis
        n_samples = f.trace[0].shape[0]
        try:
            dt_us = f.bin[segyio.BinField.Interval]
        except Exception:
            dt_us = 1000  # 1 ms fallback
        dt_ms = dt_us / 1000.0
        twt_ms = np.arange(n_samples, dtype=np.float32) * dt_ms

        x2d, y2d = _try_read_xy_maps(f, len(ilines), len(xlines))

    return SeisGrid(
        ilines=ilines,
        xlines=xlines,
        twt_ms=twt_ms,
        dt_ms=dt_ms,
        x2d=x2d,
        y2d=y2d,
    )


def read_segy_cube(segy_path: str) -> np.ndarray:
    """Return cube: [IL, XL, T] float32"""
    with segyio.open(segy_path, "r", ignore_geometry=False) as f:
        f.mmap()
        cube = segyio.tools.cube(f).astype(np.float32)
    return cube


# -------------------------
# Horizon IO (your txt format: X Y INLINE XLINE TWT_MS)
# -------------------------
def read_horizon_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    df.columns = [c.strip().upper() for c in df.columns]
    need = {"X", "Y", "INLINE", "XLINE", "TWT_MS"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Horizon file missing columns: {missing}. Got: {df.columns.tolist()}")
    return df


def horizon_to_grid(df: pd.DataFrame, grid: SeisGrid) -> np.ndarray:
    il_to_i = {int(il): i for i, il in enumerate(grid.ilines)}
    xl_to_j = {int(xl): j for j, xl in enumerate(grid.xlines)}
    H = np.full((len(grid.ilines), len(grid.xlines)), np.nan, dtype=np.float32)

    for _, r in df.iterrows():
        il = int(r["INLINE"])
        xl = int(r["XLINE"])
        if il in il_to_i and xl in xl_to_j:
            H[il_to_i[il], xl_to_j[xl]] = float(r["TWT_MS"])

    # simple interpolation fill if needed
    if np.isnan(H).any():
        for i in range(H.shape[0]):
            row = H[i]
            m = np.isnan(row)
            if (~m).sum() >= 2:
                row[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), row[~m])
                H[i] = row
        for j in range(H.shape[1]):
            col = H[:, j]
            m = np.isnan(col)
            if (~m).sum() >= 2:
                col[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), col[~m])
                H[:, j] = col

    if np.isnan(H).any():
        raise ValueError("Horizon grid still has NaNs after fill. Please check horizon coverage.")
    return H


def load_horizons(horizon_dir: str, grid: SeisGrid) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in [1, 2, 3, 4]:
        p = os.path.join(horizon_dir, f"seismic_horizon_Layer_{k}.txt")
        df = read_horizon_txt(p)
        out[f"Layer_{k}"] = horizon_to_grid(df, grid)
    return out


# -------------------------
# Zonation IO
# -------------------------
def read_zonation(zonation_path: str) -> pd.DataFrame:
    lines: List[str] = []
    with open(zonation_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("DEPTH") or s.upper().startswith("NULL"):
                continue
            lines.append(s)

    header = lines[0].split()
    data = [r.split() for r in lines[1:]]
    df = pd.DataFrame(data, columns=header)

    for c in header[1:]:
        df[c] = df[c].astype(float)

    df.columns = [c.strip().upper() for c in df.columns]
    if "WELLNAME" not in df.columns:
        df.rename(columns={df.columns[0]: "WELLNAME"}, inplace=True)
    return df


# -------------------------
# Build constraints (Section 2.1)
# -------------------------
def stratigraphic_coordinate(
    twt_ms: np.ndarray, top2d: np.ndarray, bot2d: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    t3 = twt_ms.reshape(1, 1, -1).astype(np.float32)
    top3 = top2d[:, :, None].astype(np.float32)
    bot3 = bot2d[:, :, None].astype(np.float32)
    zeta = (t3 - top3) / (bot3 - top3 + eps)
    return np.clip(zeta, 0.0, 1.0).astype(np.float32)


def interval_mask(twt_ms: np.ndarray, top2d: np.ndarray, bot2d: np.ndarray) -> np.ndarray:
    t3 = twt_ms.reshape(1, 1, -1).astype(np.float32)
    top3 = top2d[:, :, None].astype(np.float32)
    bot3 = bot2d[:, :, None].astype(np.float32)
    return ((t3 >= top3) & (t3 <= bot3)).astype(np.uint8)


def trend_descriptors(
    top2d: np.ndarray, bot2d: np.ndarray, sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_s = gaussian_filter(top2d.astype(np.float32), sigma=sigma)
    d_il, d_xl = np.gradient(top_s)
    thick = (bot2d - top2d).astype(np.float32)
    return d_il.astype(np.float32), d_xl.astype(np.float32), thick


def prior_tensor_P(zeta: np.ndarray, d_il: np.ndarray, d_xl: np.ndarray, thick: np.ndarray) -> np.ndarray:
    n_il, n_xl, n_t = zeta.shape
    g1 = np.broadcast_to(d_il[:, :, None], (n_il, n_xl, n_t)).astype(np.float32)
    g2 = np.broadcast_to(d_xl[:, :, None], (n_il, n_xl, n_t)).astype(np.float32)
    g3 = np.broadcast_to(thick[:, :, None], (n_il, n_xl, n_t)).astype(np.float32)
    return np.stack([zeta, g1, g2, g3], axis=0).astype(np.float32)  # [4,IL,XL,T]


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_u = [c.upper() for c in cols]
    for cand in candidates:
        if cand.upper() in cols_u:
            return cols[cols_u.index(cand.upper())]
    return None


def build_well_anchors_from_wellheads_and_zonation(wellheads_csv: str, zon_df: pd.DataFrame) -> pd.DataFrame:
    wh = pd.read_csv(wellheads_csv)
    wh.columns = [c.strip().upper() for c in wh.columns]
    zon_df = zon_df.copy()
    zon_df.columns = [c.strip().upper() for c in zon_df.columns]

    name_col = _pick_col(wh.columns.tolist(), ["WELLNAME", "WELL", "NAME"])
    if name_col is None:
        raise ValueError(f"Cannot find well name column in wellheads. Columns={wh.columns.tolist()}")

    x_col = _pick_col(wh.columns.tolist(), ["X", "EASTING"])
    y_col = _pick_col(wh.columns.tolist(), ["Y", "NORTHING"])
    il_col = _pick_col(wh.columns.tolist(), ["INLINE", "IL"])
    xl_col = _pick_col(wh.columns.tolist(), ["XLINE", "XL"])

    if x_col is None or y_col is None:
        if il_col is None or xl_col is None:
            raise ValueError(
                "Wellheads CSV must contain either (X,Y) or (INLINE,XLINE). "
                f"Columns={wh.columns.tolist()}"
            )

    wh = wh.rename(columns={name_col: "WELLNAME"})
    if "WELLNAME" not in zon_df.columns:
        zon_df = zon_df.rename(columns={zon_df.columns[0]: "WELLNAME"})

    df = pd.merge(wh, zon_df, on="WELLNAME", how="inner")

    layer_cols = []
    for k in [1, 2, 3, 4]:
        c = _pick_col(df.columns.tolist(), [f"LAYER_{k}", f"Layer_{k}", f"layer_{k}"])
        if c is None:
            raise ValueError(f"Cannot find Layer_{k} column in merged table. Columns={df.columns.tolist()}")
        layer_cols.append(c)

    rows = []
    for _, r in df.iterrows():
        w = str(r["WELLNAME"])
        x = float(r[x_col]) if x_col else np.nan
        y = float(r[y_col]) if y_col else np.nan
        il = int(r[il_col]) if il_col else None
        xl = int(r[xl_col]) if xl_col else None
        for c in layer_cols:
            rows.append((w, x, y, il, xl, float(r[c])))

    out = pd.DataFrame(rows, columns=["WELLNAME", "X", "Y", "INLINE", "XLINE", "TWT_MS"])
    return out


def reliability_from_anchors(
    grid: SeisGrid, anchors: pd.DataFrame, sigma_xy_m: float = 500.0, sigma_t_ms: float = 15.0
) -> np.ndarray:
    n_il, n_xl, n_t = len(grid.ilines), len(grid.xlines), len(grid.twt_ms)

    if grid.x2d is not None and grid.y2d is not None:
        X2, Y2 = grid.x2d.astype(np.float32), grid.y2d.astype(np.float32)
    else:
        dx = dy = 25.0
        xx = np.arange(n_xl, dtype=np.float32) * dx
        yy = np.arange(n_il, dtype=np.float32) * dy
        X2 = np.repeat(xx.reshape(1, -1), n_il, axis=0)
        Y2 = np.repeat(yy.reshape(-1, 1), n_xl, axis=1)

    anchors = anchors.copy()
    anchors["WELLNAME"] = anchors["WELLNAME"].astype(str)

    wells_xy = anchors.dropna(subset=["X", "Y"]).groupby("WELLNAME")[["X", "Y"]].first().reset_index()
    if len(wells_xy) == 0:
        raise ValueError("No (X,Y) found in anchors. Please ensure wellheads.csv has X and Y columns.")

    wnames = wells_xy["WELLNAME"].tolist()
    wxy = wells_xy[["X", "Y"]].values.astype(np.float32)

    tree = cKDTree(wxy)
    q = np.stack([X2.ravel(), Y2.ravel()], axis=1)
    dist_xy, idx = tree.query(q, k=1)
    dist_xy = dist_xy.astype(np.float32).reshape(n_il, n_xl)
    idx = idx.reshape(n_il, n_xl)

    marker_times_by_well: Dict[str, np.ndarray] = {}
    for w in wnames:
        ts = anchors.loc[anchors["WELLNAME"] == w, "TWT_MS"].values.astype(np.float32)
        marker_times_by_well[w] = np.sort(ts)

    t_axis = grid.twt_ms.astype(np.float32)
    C = np.zeros((n_il, n_xl, n_t), dtype=np.float32)

    for i in range(n_il):
        for j in range(n_xl):
            w = wnames[int(idx[i, j])]
            ts = marker_times_by_well[w]
            k = np.searchsorted(ts, t_axis)
            k0 = np.clip(k - 1, 0, len(ts) - 1)
            k1 = np.clip(k, 0, len(ts) - 1)
            dt = np.minimum(np.abs(t_axis - ts[k0]), np.abs(t_axis - ts[k1]))
            d2 = (dist_xy[i, j] / sigma_xy_m) ** 2 + (dt / sigma_t_ms) ** 2
            C[i, j, :] = np.exp(-d2).astype(np.float32)

    return C


def build_constraints(
    paths: DataPaths,
    sigma_trend: float = 3.0,
    sigma_xy_m: float = 500.0,
    sigma_t_ms: float = 15.0,
) -> Dict[str, np.ndarray]:
    os.makedirs(paths.processed_dir, exist_ok=True)

    grid = read_segy_grid(paths.segy_seis)
    horizons = load_horizons(paths.seismic_horizon_dir, grid)

    top2d = horizons["Layer_1"]
    bot2d = horizons["Layer_4"]

    zeta = stratigraphic_coordinate(grid.twt_ms, top2d, bot2d)
    M = interval_mask(grid.twt_ms, top2d, bot2d)

    d_il, d_xl, thick = trend_descriptors(top2d, bot2d, sigma=sigma_trend)
    P = prior_tensor_P(zeta, d_il, d_xl, thick)

    zon = read_zonation(paths.zonation_txt)
    anchors = build_well_anchors_from_wellheads_and_zonation(paths.wellheads_csv, zon)

    C = reliability_from_anchors(grid, anchors, sigma_xy_m=sigma_xy_m, sigma_t_ms=sigma_t_ms)

    return {
        "P": P,  # [4,IL,XL,T]
        "C": C,  # [IL,XL,T]
        "M": M,  # [IL,XL,T]
        "top2d_ms": top2d,
        "bot2d_ms": bot2d,
        "twt_ms": grid.twt_ms,
        "ilines": grid.ilines,
        "xlines": grid.xlines,
        "dt_ms": np.float32(grid.dt_ms),
    }


def save_constraints_npz(out_path: str, pack: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(out_path, **pack)