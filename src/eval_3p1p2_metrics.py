# eval_3p1p2_metrics.py
# -*- coding: utf-8 -*-

import os, sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ---------------------------
# 0) Path setup
# ---------------------------
nb_dir = Path.cwd()
repo_root = nb_dir  # 如果你在 repo 根目录运行
# 如果你在 notebooks 目录运行，把上一行改成：repo_root = nb_dir.parent
sys.path.insert(0, str(repo_root))

print("CWD:", nb_dir)
print("Repo root:", repo_root)

# ---------------------------
# 1) Imports from your repo
# ---------------------------
from src.geo_constraints import DataPaths
from src.dataset_vie import StanfordVIEWellPatchDataset
from src.models.geo_cnn_multitask import GeoCNNMultiTask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------------------
# 2) User config
# ---------------------------
DATA_ROOT = r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data"
paths = DataPaths(DATA_ROOT)

constraints_npz = os.path.join(paths.processed_dir, "constraints.npz")
assert os.path.isfile(constraints_npz), f"Missing constraints_npz: {constraints_npz}"

# dataset (must align with training)
ds = StanfordVIEWellPatchDataset(
    paths,
    constraints_npz,
    patch_hw=4,
    use_masked_y=True,
    normalize=True
)

print("processed_dir:", paths.processed_dir)
print("constraints_npz:", constraints_npz)
print("Dataset size:", len(ds))
print("AI mean/std:", ds.ai_mean, ds.ai_std)
print("Seis mean/std:", ds.seis_mean, ds.seis_std)

# splits
split_dir = os.path.join(paths.processed_dir, "splits")
os.makedirs(split_dir, exist_ok=True)
train_f = os.path.join(split_dir, "train_idx.npy")
val_f   = os.path.join(split_dir, "val_idx.npy")
test_f  = os.path.join(split_dir, "test_idx.npy")

def make_splits(n, seed=2026, frac_train=0.8, frac_val=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int32)
    rng.shuffle(idx)
    n_train = int(frac_train * n)
    n_val   = int(frac_val   * n)
    train = idx[:n_train]
    val   = idx[n_train:n_train+n_val]
    test  = idx[n_train+n_val:]
    return train, val, test

if os.path.isfile(train_f) and os.path.isfile(val_f) and os.path.isfile(test_f):
    train_idx = np.load(train_f)
    val_idx   = np.load(val_f)
    test_idx  = np.load(test_f)
    print("Loaded splits:", split_dir)
else:
    train_idx, val_idx, test_idx = make_splits(len(ds), seed=2026)
    np.save(train_f, train_idx)
    np.save(val_f, val_idx)
    np.save(test_f, test_idx)
    print("Created splits:", split_dir)

print("train:", len(train_idx), "val:", len(val_idx), "test:", len(test_idx))

# checkpoints
ckpt_dir_final = os.path.join(paths.processed_dir, "checkpoints_multitask_final")
ckpt_joint = os.path.join(ckpt_dir_final, "best_joint.pt")
ckpt_ai    = os.path.join(ckpt_dir_final, "best_ai.pt")
assert os.path.isfile(ckpt_joint), f"Missing: {ckpt_joint}"
assert os.path.isfile(ckpt_ai),    f"Missing: {ckpt_ai}"
print("ckpt_joint:", ckpt_joint)
print("ckpt_ai   :", ckpt_ai)

# output
OUT_DIR = os.path.join(paths.processed_dir, "eval_reports")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "metrics_3p1p2_test.json")
OUT_CSV  = os.path.join(OUT_DIR, "metrics_3p1p2_test.csv")

# ---------------------------
# 3) Optional physics forward
# ---------------------------
try:
    from src.physics_forward import forward_seismic_from_ai
    HAS_PHYS = True
    print("Physics forward: OK")
except Exception as e:
    HAS_PHYS = False
    print("Physics forward: NOT available -> will skip L2/NRMS. Error:", repr(e))

# ---------------------------
# 4) Model build/load
# ---------------------------
def build_model():
    # MUST match training
    return GeoCNNMultiTask(
        in_channels=7,
        base=32,
        t=200,
        n_facies=4
    ).to(device)

def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return ckpt

def unpack_outputs(out):
    """
    returns: ai_pred, facies_logits (optional)
    """
    ai_pred = None
    facies_logits = None
    if isinstance(out, dict):
        for k in ["ai", "ai_pred", "y", "imp", "impedance"]:
            if k in out:
                ai_pred = out[k]; break
        for k in ["facies", "facies_logits", "logits", "facies_logit"]:
            if k in out:
                facies_logits = out[k]; break
    elif isinstance(out, (list, tuple)):
        if len(out) >= 1: ai_pred = out[0]
        if len(out) >= 2: facies_logits = out[1]
    else:
        ai_pred = out
    return ai_pred, facies_logits

def mask_center_trace(m_patch):
    """
    m_patch: [1,H,W,T] or [B,1,H,W,T]
    """
    if m_patch.ndim == 4:
        _, H, W, T = m_patch.shape
        return m_patch[0, H//2, W//2, :]
    elif m_patch.ndim == 5:
        B, _, H, W, T = m_patch.shape
        return m_patch[:, 0, H//2, W//2, :]
    else:
        raise ValueError(f"Unexpected m_patch shape: {m_patch.shape}")

m_joint = build_model()
m_ai    = build_model()
_ = load_ckpt(m_joint, ckpt_joint)
_ = load_ckpt(m_ai, ckpt_ai)
print("Loaded best_joint & best_ai")

# ---------------------------
# 5) 3.1.2 metrics implementations
# ---------------------------
EPS = 1e-12

def _flat_apply_mask(y_pred, y_true, mask=None):
    yp = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    yt = np.asarray(y_true, dtype=np.float32).reshape(-1)
    if mask is None:
        m = np.ones_like(yt, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool).reshape(-1)
    yp = yp[m]
    yt = yt[m]
    return yp, yt

def MAE(y_pred, y_true, mask=None):
    yp, yt = _flat_apply_mask(y_pred, y_true, mask)
    return float(np.mean(np.abs(yp - yt)) if yp.size else np.nan)

def RMSE(y_pred, y_true, mask=None):
    yp, yt = _flat_apply_mask(y_pred, y_true, mask)
    return float(np.sqrt(np.mean((yp - yt) ** 2)) if yp.size else np.nan)

def RE(y_pred, y_true, mask=None, eps=EPS):
    yp, yt = _flat_apply_mask(y_pred, y_true, mask)
    return float(np.mean(np.abs(yp - yt) / (np.abs(yt) + eps)) if yp.size else np.nan)

def PearsonR(y_pred, y_true, mask=None, eps=EPS):
    yp, yt = _flat_apply_mask(y_pred, y_true, mask)
    if yp.size < 2:
        return float("nan")
    yp0 = yp - yp.mean()
    yt0 = yt - yt.mean()
    denom = (np.sqrt(np.sum(yp0**2)) * np.sqrt(np.sum(yt0**2)) + eps)
    return float(np.sum(yp0 * yt0) / denom)

def SSIM_global_2d(x, y, beta1=1e-4, beta2=9e-4, mask=None):
    """
    SSIM(x,y) = (2*mu_x*mu_y+beta1)(2*sigma_xy+beta2) / ((mu_x^2+mu_y^2+beta1)(sigma_x^2+sigma_y^2+beta2))
    computed on 2D arrays
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    assert x.shape == y.shape, f"SSIM shape mismatch: {x.shape} vs {y.shape}"
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        xv = x[m]; yv = y[m]
    else:
        xv = x.reshape(-1); yv = y.reshape(-1)
    if xv.size < 2:
        return float("nan")

    mu_x = float(np.mean(xv))
    mu_y = float(np.mean(yv))
    sig_x = float(np.std(xv, ddof=0))
    sig_y = float(np.std(yv, ddof=0))
    sig_xy = float(np.mean((xv - mu_x) * (yv - mu_y)))

    num = (2 * mu_x * mu_y + beta1) * (2 * sig_xy + beta2)
    den = (mu_x**2 + mu_y**2 + beta1) * (sig_x**2 + sig_y**2 + beta2)
    return float(num / (den + EPS))

def L2_misfit_norm(d_syn, d_obs, mask=None, eps=EPS):
    dsyn = np.asarray(d_syn, dtype=np.float32).reshape(-1)
    dobs = np.asarray(d_obs, dtype=np.float32).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        dsyn = dsyn[m]; dobs = dobs[m]
    if dsyn.size == 0:
        return float("nan")
    return float(np.linalg.norm(dsyn - dobs) / (np.linalg.norm(dobs) + eps))

def RMS(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean(x**2)) if x.size else np.nan)

def NRMS(d_syn, d_obs, mask=None, eps=EPS):
    dsyn = np.asarray(d_syn, dtype=np.float32)
    dobs = np.asarray(d_obs, dtype=np.float32)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        dsyn = dsyn[m]; dobs = dobs[m]
    if dsyn.size == 0:
        return float("nan")
    num = 2.0 * RMS(dsyn - dobs)
    den = (RMS(dsyn) + RMS(dobs) + eps)
    return float((num / den) * 100.0)

# ---------------------------
# 6) Collect predictions on test set (center trace)
# ---------------------------
@torch.no_grad()
def collect_center_traces(model, indices, name="model"):
    """
    Returns dict with concatenated arrays (denorm):
      ai_true_den: [N,T]
      ai_pred_den: [N,T]
      mask:        [N,T]  bool
      seis_obs_den:[N,T]  (from input seismic center trace, denorm)
      seis_syn_den:[N,T]  (forward from ai_pred_den) or None
      il,xl,wellname lists
    """
    ai_true_den = []
    ai_pred_den = []
    mask_all    = []
    seis_obs_den_all = []
    seis_syn_den_all = [] if HAS_PHYS else None

    il_list, xl_list, wn_list = [], [], []

    for idx in indices.tolist():
        b = ds[int(idx)]
        x = b["x"][None].to(device)  # [1,1,H,W,T] norm seismic patch
        p = b["p"][None].to(device)
        c = b["c"][None].to(device)
        m = b["m"][None].to(device)

        out = model(x, p, c, m)
        ai_pred, _ = unpack_outputs(out)
        ai_pred = ai_pred.squeeze()
        if ai_pred.ndim != 1:
            ai_pred = ai_pred.reshape(-1)  # [T]

        y = b["y"].float()  # [T] norm
        mc = mask_center_trace(b["m"]).float()  # [T] 0/1
        mask = (mc.numpy() > 0.5)  # bool [T]

        # denorm AI
        y_den = (y.numpy() * float(ds.ai_std) + float(ds.ai_mean)).astype(np.float32)
        p_den = (ai_pred.detach().cpu().numpy() * float(ds.ai_std) + float(ds.ai_mean)).astype(np.float32)

        # observed seismic center trace denorm (from input x patch)
        x_patch = b["x"].float()  # [1,H,W,T] norm
        H = x_patch.shape[1]; W = x_patch.shape[2]
        seis_obs_norm = x_patch[0, H//2, W//2, :].numpy().astype(np.float32)
        seis_obs_den  = seis_obs_norm * float(ds.seis_std) + float(ds.seis_mean)

        ai_true_den.append(y_den)
        ai_pred_den.append(p_den)
        mask_all.append(mask)
        seis_obs_den_all.append(seis_obs_den)

        # forward seismic (optional)
        if HAS_PHYS:
            ai_den_t = torch.from_numpy(p_den).to(device).float()[None, :]  # [1,T]
            try:
                seis_syn = forward_seismic_from_ai(ai_den_t)  # expected [1,T] or [T]
                seis_syn = seis_syn.squeeze().detach().cpu().numpy().astype(np.float32).reshape(-1)
            except Exception:
                seis_syn = np.full_like(seis_obs_den, np.nan, dtype=np.float32)
            seis_syn_den_all.append(seis_syn)

        il_list.append(int(b["il"]))
        xl_list.append(int(b["xl"]))
        wn_list.append(str(b["wellname"]))

    out = {
        "name": name,
        "ai_true_den": np.stack(ai_true_den, axis=0),   # [N,T]
        "ai_pred_den": np.stack(ai_pred_den, axis=0),   # [N,T]
        "mask":        np.stack(mask_all, axis=0),      # [N,T] bool
        "seis_obs_den":np.stack(seis_obs_den_all, axis=0), # [N,T]
        "il": il_list,
        "xl": xl_list,
        "wellname": wn_list
    }
    if HAS_PHYS:
        out["seis_syn_den"] = np.stack(seis_syn_den_all, axis=0)  # [N,T]
    else:
        out["seis_syn_den"] = None
    return out

# ---------------------------
# 7) Compute 3.1.2 report
# ---------------------------
def compute_3p1p2_metrics(pack):
    """
    pack: output of collect_center_traces()
    returns dict of metrics required by 3.1.2
    """
    y = pack["ai_true_den"]  # [N,T]
    p = pack["ai_pred_den"]  # [N,T]
    m = pack["mask"]         # [N,T] bool
    N, T = y.shape

    # impedance metrics (masked)
    metrics = {}
    metrics["n_traces"] = int(N)
    metrics["n_samples_masked"] = int(np.sum(m))
    metrics["AI_MAE_masked"] = MAE(p, y, m)
    metrics["AI_RMSE_masked"] = RMSE(p, y, m)
    metrics["AI_RE_masked"] = RE(p, y, m)
    metrics["AI_R_masked"] = PearsonR(p, y, m)

    # impedance metrics (all samples)
    metrics["AI_MAE_all"] = MAE(p, y, None)
    metrics["AI_RMSE_all"] = RMSE(p, y, None)
    metrics["AI_RE_all"] = RE(p, y, None)
    metrics["AI_R_all"] = PearsonR(p, y, None)

    # SSIM on a 2D "section" made by stacking center traces
    # (sorted by (il,xl) for reproducibility): image shape [T, N]
    order = np.lexsort((np.asarray(pack["xl"]), np.asarray(pack["il"])))
    y_sec = y[order].T  # [T,N]
    p_sec = p[order].T  # [T,N]
    m_sec = m[order].T  # [T,N]

    metrics["AI_SSIM_section_masked"] = SSIM_global_2d(p_sec, y_sec, beta1=1e-4, beta2=9e-4, mask=m_sec)
    metrics["AI_SSIM_section_all"] = SSIM_global_2d(p_sec, y_sec, beta1=1e-4, beta2=9e-4, mask=None)

    # seismic-domain physical consistency
    if pack["seis_syn_den"] is not None:
        d_obs = pack["seis_obs_den"]        # [N,T]
        d_syn = pack["seis_syn_den"]        # [N,T]
        # You can choose mask in seismic domain:
        # Option A: use same mask as impedance (common for fair interval comparison)
        # Option B: no mask (full wavelet window)
        metrics["Seis_L2_masked"] = L2_misfit_norm(d_syn, d_obs, m)
        metrics["Seis_NRMS_masked(%)"] = NRMS(d_syn, d_obs, m)

        metrics["Seis_L2_all"] = L2_misfit_norm(d_syn, d_obs, None)
        metrics["Seis_NRMS_all(%)"] = NRMS(d_syn, d_obs, None)
    else:
        metrics["Seis_L2_masked"] = None
        metrics["Seis_NRMS_masked(%)"] = None
        metrics["Seis_L2_all"] = None
        metrics["Seis_NRMS_all(%)"] = None

    return metrics

# ---------------------------
# 8) Run on test set: best_ai vs best_joint
# ---------------------------
print("\nCollecting test predictions...")
pack_ai    = collect_center_traces(m_ai,    test_idx, name="best_ai")
pack_joint = collect_center_traces(m_joint, test_idx, name="best_joint")

print("Computing metrics...")
met_ai    = compute_3p1p2_metrics(pack_ai)
met_joint = compute_3p1p2_metrics(pack_joint)

# Combine into a single report
report = {
    "data_root": DATA_ROOT,
    "processed_dir": paths.processed_dir,
    "constraints_npz": constraints_npz,
    "ckpt_dir": ckpt_dir_final,
    "ckpt_best_ai": ckpt_ai,
    "ckpt_best_joint": ckpt_joint,
    "physics_forward_available": bool(HAS_PHYS),
    "metrics_3p1p2": {
        "best_ai": met_ai,
        "best_joint": met_joint
    }
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("Saved JSON:", OUT_JSON)

# Also save a CSV-friendly table (one row per model)
try:
    import pandas as pd
    rows = []
    for name, met in report["metrics_3p1p2"].items():
        row = {"model": name}
        row.update(met)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("Saved CSV :", OUT_CSV)
    print(df)
except Exception as e:
    print("pandas not available or CSV save failed:", repr(e))
    print("Metrics (best_ai):", met_ai)
    print("Metrics (best_joint):", met_joint)

print("\nDone.")