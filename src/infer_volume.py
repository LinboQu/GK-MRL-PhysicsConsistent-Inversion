from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import segyio

from src.geo_constraints import DataPaths, read_segy_cube, read_segy_grid
from src.models.geo_cnn_multitask import GeoCNNMultiTask


# -------------------------
# helpers: patch extraction
# -------------------------
def _pad_edge_2d_time(cube: np.ndarray, pad: int) -> np.ndarray:
    """
    cube: [IL,XL,T]
    pad along IL/XL with edge mode
    """
    if pad <= 0:
        return cube
    return np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="edge")


def extract_patch(cube_pad: np.ndarray, i: int, j: int, pad: int) -> np.ndarray:
    """
    cube_pad: [IL+2p, XL+2p, T]
    i,j are indices in original cube
    return patch [H,W,T] with H=W=2p+1
    """
    ii = i + pad
    jj = j + pad
    return cube_pad[ii - pad: ii + pad + 1, jj - pad: jj + pad + 1, :]


# -------------------------
# SEG-Y writer (copy geometry from template)
# -------------------------
def write_segy_from_template(
    template_segy: str,
    out_segy: str,
    cube_il_xl_t: np.ndarray,
    dtype: str = "float32",
    format_code: int = 5,
) -> None:
    """
    Write a 3D cube [IL,XL,T] into SEG-Y using template geometry/headers.
    format_code:
      5 = IEEE float
      1 = IBM float (avoid)
      3 = int16 (some segyio builds use 3 for int16; but safest is still 5 with float)
    We'll set samples/ilines/xlines from template.
    """
    cube = cube_il_xl_t
    assert cube.ndim == 3, "cube must be [IL,XL,T]"

    with segyio.open(template_segy, "r", ignore_geometry=False) as f:
        f.mmap()

        ilines = list(map(int, f.ilines))
        xlines = list(map(int, f.xlines))
        samples = np.array(f.samples, dtype=np.int32)  # usually in ms

        # Build spec
        spec = segyio.spec()
        spec.ilines = ilines
        spec.xlines = xlines
        spec.samples = samples
        spec.sorting = f.sorting
        spec.format = format_code  # 5 = IEEE float32

        os.makedirs(os.path.dirname(out_segy), exist_ok=True)
        if os.path.exists(out_segy):
            os.remove(out_segy)

        with segyio.create(out_segy, spec) as g:
            # copy binary header (interval etc.)
            g.bin = f.bin

            # copy per-trace headers (XY, inline/xline, etc.)
            for tr in range(f.tracecount):
                g.header[tr] = f.header[tr]

            # write trace data: segyio expects trace-major order
            # Template cube ordering from segyio.tools.cube is [IL,XL,T]
            # Trace index order is the same as flattening cube in C order over IL then XL.
            data = cube.astype(np.float32 if dtype == "float32" else np.int16)
            k = 0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    g.trace[k] = data[i, j, :]
                    k += 1

            g.flush()


# -------------------------
# main volume inference
# -------------------------
@dataclass
class InferConfig:
    data_root: str
    ckpt_path: str
    patch_hw: int = 4
    batch_size: int = 32
    device: str = "cuda"
    # outputs
    out_dir_name: str = "inversion_volume"


def main():
    cfg = InferConfig(
        data_root=r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data",
        ckpt_path=r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data\processed\checkpoints_multitask_facies_warmup_fix\best_joint.pt",
        patch_hw=4,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_dir_name="inversion_volume",
    )

    paths = DataPaths(cfg.data_root)

    # --- load seismic + constraints ---
    print("Loading seismic cube:", paths.segy_seis)
    seis = read_segy_cube(paths.segy_seis)  # [IL,XL,T] float32

    print("Loading constraints:", os.path.join(paths.processed_dir, "constraints.npz"))
    pack = np.load(os.path.join(paths.processed_dir, "constraints.npz"))
    P = pack["P"].astype(np.float32)          # [4,IL,XL,T]
    C = pack["C"].astype(np.float32)          # [IL,XL,T]
    M = pack["M"].astype(np.float32)          # [IL,XL,T]
    # Sanity
    assert seis.shape == C.shape == M.shape, f"Shape mismatch: seis{seis.shape}, C{C.shape}, M{M.shape}"
    assert P.shape[1:] == seis.shape, f"P mismatch: P{P.shape}, seis{seis.shape}"

    n_il, n_xl, n_t = seis.shape
    pad = cfg.patch_hw
    H = W = 2 * pad + 1

    # --- load model ---
    device = torch.device(cfg.device)
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    print("Loaded ckpt:", cfg.ckpt_path, "epoch:", ckpt.get("epoch"), "mode:", ckpt.get("mode"))

    model = GeoCNNMultiTask(in_channels=7, base=32, t=n_t, n_facies=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # normalization from ckpt (must match training)
    ai_mean = float(ckpt.get("ai_mean", 0.0))
    ai_std  = float(ckpt.get("ai_std", 1.0))
    seis_mean = float(ckpt.get("seis_mean", 0.0))
    seis_std  = float(ckpt.get("seis_std", 1.0))

    # --- normalize seismic (same as dataset) ---
    seis_n = (seis - seis_mean) / (seis_std + 1e-8)

    # --- pad cubes for patch extraction ---
    seis_pad = _pad_edge_2d_time(seis_n, pad)          # [IL+2p,XL+2p,T]
    C_pad    = _pad_edge_2d_time(C, pad)
    M_pad    = _pad_edge_2d_time(M, pad)

    # P is [4,IL,XL,T] -> pad each channel
    P_pad = np.stack([_pad_edge_2d_time(P[ch], pad) for ch in range(P.shape[0])], axis=0)  # [4,IL+2p,XL+2p,T]

    # --- allocate outputs ---
    ai_pred = np.zeros((n_il, n_xl, n_t), dtype=np.float32)
    fac_pred = np.full((n_il, n_xl, n_t), -1, dtype=np.int16)  # -1 outside mask

    # --- batching over spatial grid ---
    idx_list: List[Tuple[int, int]] = [(i, j) for i in range(n_il) for j in range(n_xl)]
    bs = cfg.batch_size

    print(f"Running volume inference: IL={n_il}, XL={n_xl}, T={n_t}, patch={H}x{W}, batch={bs} on {device} ...")

    with torch.no_grad():
        for s in range(0, len(idx_list), bs):
            batch_ij = idx_list[s:s + bs]

            xb = np.zeros((len(batch_ij), 1, H, W, n_t), dtype=np.float32)
            pb = np.zeros((len(batch_ij), 4, H, W, n_t), dtype=np.float32)
            cb = np.zeros((len(batch_ij), 1, H, W, n_t), dtype=np.float32)
            mb = np.zeros((len(batch_ij), 1, H, W, n_t), dtype=np.float32)

            for k, (i, j) in enumerate(batch_ij):
                xb[k, 0] = extract_patch(seis_pad, i, j, pad)
                cb[k, 0] = extract_patch(C_pad, i, j, pad)
                mb[k, 0] = extract_patch(M_pad, i, j, pad)
                for ch in range(4):
                    pb[k, ch] = extract_patch(P_pad[ch], i, j, pad)

            x_t = torch.from_numpy(xb).to(device)
            p_t = torch.from_numpy(pb).to(device)
            c_t = torch.from_numpy(cb).to(device)
            m_t = torch.from_numpy(mb).to(device)

            ai_hat_n, fac_logits = model(x_t, p_t, c_t, m_t)  # [B,T], [B,K,T]
            ai_hat_n = ai_hat_n.detach().cpu().numpy().astype(np.float32)
            fac_cls  = fac_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int16)  # [B,T]

            # denormalize AI
            ai_hat = ai_hat_n * (ai_std + 1e-8) + ai_mean

            # write back
            for k, (i, j) in enumerate(batch_ij):
                ai_pred[i, j, :] = ai_hat[k, :]
                # apply center-mask: outside interval -> -1
                m_center = mb[k, 0, pad, pad, :]  # [T]
                fac_line = fac_cls[k, :]
                fac_line[m_center < 0.5] = -1
                fac_pred[i, j, :] = fac_line

            if (s // bs) % 50 == 0:
                print(f"  progress: {s}/{len(idx_list)}")

    # also apply mask to ai_pred (optional): set outside interval to NaN for nicer visualization
    ai_pred_masked = ai_pred.copy()
    ai_pred_masked[M < 0.5] = np.nan

    out_dir = os.path.join(paths.processed_dir, cfg.out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- save NPY ---
    npy_ai = os.path.join(out_dir, "AI_pred_cube.npy")
    npy_ai_masked = os.path.join(out_dir, "AI_pred_cube_masked.npy")
    npy_fac = os.path.join(out_dir, "Facies_pred_cube.npy")
    np.save(npy_ai, ai_pred)
    np.save(npy_ai_masked, ai_pred_masked)
    np.save(npy_fac, fac_pred)
    print("Saved NPY:")
    print(" ", npy_ai)
    print(" ", npy_ai_masked)
    print(" ", npy_fac)

    # --- save SEG-Y ---
    # AI: IEEE float32
    out_ai_segy = os.path.join(out_dir, "AI_pred_cube.sgy")
    write_segy_from_template(paths.segy_seis, out_ai_segy, np.nan_to_num(ai_pred_masked, nan=0.0).astype(np.float32), dtype="float32", format_code=5)

    # Facies: store as float32 too (more compatible), values -1/0/1/2/3
    out_fac_segy = os.path.join(out_dir, "Facies_pred_cube.sgy")
    write_segy_from_template(paths.segy_seis, out_fac_segy, fac_pred.astype(np.float32), dtype="float32", format_code=5)

    print("Saved SEG-Y:")
    print(" ", out_ai_segy)
    print(" ", out_fac_segy)
    print("Done.")


if __name__ == "__main__":
    main()