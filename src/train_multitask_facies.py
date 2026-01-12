from __future__ import annotations

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.geo_constraints import DataPaths
from src.dataset_vie import StanfordVIEWellPatchDataset
from src.models.geo_cnn_multitask import GeoCNNMultiTask
from src.physics_forward import forward_seismic_from_ai
from src.utils.alpha_stats import alpha_stats
from src.utils.train_utils import set_seed, AverageMeter, ensure_dir


# -----------------------------
# losses
# -----------------------------
def weighted_masked_mse(y_hat, y, m, w=None):
    """
    y_hat, y: [B,T]
    m: [B,T] 0/1 mask
    w: [B,T] weight in [0,1] (optional)
    """
    diff2 = (y_hat - y) ** 2
    ww = m if w is None else (m * w)
    return (diff2 * ww).sum() / (ww.sum() + 1e-8)


def weighted_masked_ce(logits, y, valid, w=None, ignore_index=-1):
    """
    logits: [B,K,T]
    y: [B,T] int64 in {0..K-1} or -1
    valid: [B,T] 0/1
    w: [B,T] weight in [0,1] (optional)
    """
    B, K, T = logits.shape
    loss_bt = F.cross_entropy(
        logits.permute(0, 2, 1).reshape(-1, K),  # [B*T,K]
        y.reshape(-1),                           # [B*T]
        reduction="none",
        ignore_index=ignore_index
    ).reshape(B, T)

    ww = valid if w is None else (valid * w)
    return (loss_bt * ww).sum() / (ww.sum() + 1e-8)


# -----------------------------
# one epoch
# -----------------------------
def run_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optim: torch.optim.Optimizer | None,
    facies_on: bool,
    lam_ai: float,
    lam_phys: float,
    lam_fac: float,
    cc_facies_thresh: float,
    dt_ms: float,
    f0_hz: float,
    wavelet_nt: int,
    epoch: int,
    print_alpha: bool,
    print_alpha_every: int,
) -> dict:
    is_train = optim is not None
    model.train(is_train)

    mt = AverageMeter()
    mai = AverageMeter()
    mph = AverageMeter()
    mfa = AverageMeter()

    alpha_log = ""

    for batch_i, batch in enumerate(loader):
        x = batch["x"].to(device)
        p = batch["p"].to(device)
        c = batch["c"].to(device)
        m = batch["m"].to(device)

        y_ai = batch["y"].to(device)              # [B,T]
        cc = batch["c_center"].to(device)         # [B,T]

        y_fac = batch["facies_center"].to(device)
        v_fac = batch["facies_valid"].to(device)

        # center trace mask & observed seismic
        H, W = m.shape[2], m.shape[3]
        mc = m[:, 0, H // 2, W // 2, :]           # [B,T]
        s_obs = x[:, 0, H // 2, W // 2, :]        # [B,T]

        if is_train:
            optim.zero_grad(set_to_none=True)

        # forward
        ai_hat, fac_logits = model(x, p, c, m)

        if print_alpha and is_train and batch_i == 0 and (epoch % print_alpha_every == 0):
            alphas = None

            if hasattr(model, "get_alphas") and callable(model.get_alphas):
                alphas = model.get_alphas()

            if not alphas and hasattr(model, "_last_alphas"):
                alphas = model._last_alphas

            if not alphas:
                alpha_log = f"[alpha][epoch {epoch}] (no alphas found on model; skip)"
            else:
                lines = [f"[alpha][epoch {epoch}] keys={list(alphas.keys())}"]
                for k, a in alphas.items():
                    m, x, h = alpha_stats(a)
                    lines.append(
                        f"[alpha] {k}: shape={tuple(a.shape)} "
                        f"mean={np.round(m,3)} max={np.round(x,3)} H={h:.3f}"
                    )
                alpha_log = "\n".join(lines)

        # losses
        L_ai = weighted_masked_mse(ai_hat, y_ai, mc, w=cc)

        s_syn = forward_seismic_from_ai(ai_hat, dt_ms=dt_ms, f0_hz=f0_hz, wavelet_nt=wavelet_nt)
        L_phys = weighted_masked_mse(s_syn, s_obs, mc, w=cc)

        if not facies_on:
            L_fac = torch.tensor(0.0, device=device)
            lam_fac_eff = 0.0
        else:
            cc_gate = (cc > cc_facies_thresh).float()
            L_fac = weighted_masked_ce(
                fac_logits,
                y_fac,
                valid=v_fac * cc_gate,
                w=cc,
                ignore_index=-1
            )
            lam_fac_eff = lam_fac

        loss = lam_ai * L_ai + lam_phys * L_phys + lam_fac_eff * L_fac

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        mt.update(loss.item(), x.size(0))
        mai.update(L_ai.item(), x.size(0))
        mph.update(L_phys.item(), x.size(0))
        mfa.update(L_fac.item(), x.size(0))

    out = {
        "total": mt.avg,
        "ai": mai.avg,
        "phys": mph.avg,
        "fac": mfa.avg,
    }
    if alpha_log:
        out["alpha_log"] = alpha_log
    return out


# -----------------------------
# main
# -----------------------------
def main():
    set_seed(2026)

    DATA_ROOT = r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data"
    paths = DataPaths(DATA_ROOT)

    constraints_npz = os.path.join(paths.processed_dir, "constraints.npz")
    split_dir = os.path.join(paths.processed_dir, "splits")
    train_idx = np.load(os.path.join(split_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(split_dir, "val_idx.npy"))

    ds = StanfordVIEWellPatchDataset(
        paths=paths,
        constraints_npz=constraints_npz,
        patch_hw=4,
        use_masked_y=True,
        normalize=True,
    )

    train_set = Subset(ds, train_idx.tolist())
    val_set = Subset(ds, val_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCNNMultiTask(in_channels=7, base=32, t=200, n_facies=4).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # IMPORTANT: new dir to avoid mixing old ckpts
    out_dir = os.path.join(paths.processed_dir, "checkpoints_multitask_final")
    ensure_dir(out_dir)

    # loss weights
    lam_ai = 1.0
    lam_phys = 0.3
    lam_fac = 0.5

    # warmup & gating
    facies_warmup_epochs = 8
    cc_facies_thresh = 0.35

    # forward params
    dt_ms = 1.0
    f0_hz = 40.0
    wavelet_nt = 81

    # debug switches
    PRINT_ALPHA = True          # only prints once per epoch (train first batch)
    PRINT_ALPHA_EVERY = 5
    MAX_EPOCHS = 50

    # best trackers
    best_ai = float("inf")       # best val_ai (typically during warmup)
    best_joint = float("inf")    # best val_total (only after facies_on)

    # early stopping only for joint stage
    patience = 8
    bad = 0

    history = {
        "train_total": [], "val_total": [],
        "train_ai": [], "val_ai": [],
        "train_phys": [], "val_phys": [],
        "train_fac": [], "val_fac": [],
        "best_ai": [], "best_joint": [],
        "lr": []
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        facies_on = (epoch > facies_warmup_epochs)

        # -------------------------
        # train
        # -------------------------
        tr = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optim=optim,
            facies_on=facies_on,
            lam_ai=lam_ai,
            lam_phys=lam_phys,
            lam_fac=lam_fac,
            cc_facies_thresh=cc_facies_thresh,
            dt_ms=dt_ms,
            f0_hz=f0_hz,
            wavelet_nt=wavelet_nt,
            epoch=epoch,
            print_alpha=PRINT_ALPHA,
            print_alpha_every=PRINT_ALPHA_EVERY,
        )

        # -------------------------
        # val
        # -------------------------
        with torch.no_grad():
            va = run_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optim=None,
                facies_on=facies_on,
                lam_ai=lam_ai,
                lam_phys=lam_phys,
                lam_fac=lam_fac,
                cc_facies_thresh=cc_facies_thresh,
                dt_ms=dt_ms,
                f0_hz=f0_hz,
                wavelet_nt=wavelet_nt,
                epoch=epoch,
                print_alpha=False,   # never print alpha in val to avoid noise
                print_alpha_every=PRINT_ALPHA_EVERY,
            )

        train_total, train_ai, train_phys, train_fac = tr["total"], tr["ai"], tr["phys"], tr["fac"]
        val_total, val_ai, val_phys, val_fac = va["total"], va["ai"], va["phys"], va["fac"]
        
        if "alpha_log" in tr:
            print(tr["alpha_log"])

        # -------------------------
        # log
        # -------------------------
        lr_now = optim.param_groups[0]["lr"]
        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_ai"].append(train_ai)
        history["val_ai"].append(val_ai)
        history["train_phys"].append(train_phys)
        history["val_phys"].append(val_phys)
        history["train_fac"].append(train_fac)
        history["val_fac"].append(val_fac)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:02d} | "
            f"train {train_total:.6f} (ai {train_ai:.4f}, phys {train_phys:.4f}, fac {train_fac:.4f}) | "
            f"val {val_total:.6f} (ai {val_ai:.4f}, phys {val_phys:.4f}, fac {val_fac:.4f}) | "
            f"facies_on={facies_on} | lr={lr_now:.2e}"
        )

        scheduler.step(val_total)

        # -------------------------
        # checkpointing logic
        # -------------------------
        # (A) best_ai: allow update anytime
        if val_ai < best_ai:
            best_ai = val_ai
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "mode": "best_ai",
                    "best_ai": best_ai,
                    "lam_ai": lam_ai, "lam_phys": lam_phys, "lam_fac": lam_fac,
                    "facies_warmup_epochs": facies_warmup_epochs,
                    "cc_facies_thresh": cc_facies_thresh,
                    "dt_ms": dt_ms, "f0_hz": f0_hz, "wavelet_nt": wavelet_nt,
                    "ai_mean": ds.ai_mean, "ai_std": ds.ai_std,
                    "seis_mean": ds.seis_mean, "seis_std": ds.seis_std,
                },
                os.path.join(out_dir, "best_ai.pt")
            )
            print(f"  saved: best_ai.pt (best_ai={best_ai:.6f})")

        # (B) best_joint: ONLY after facies_on, use val_total
        if facies_on:
            if val_total < best_joint:
                best_joint = val_total
                bad = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optim_state": optim.state_dict(),
                        "mode": "best_joint",
                        "best_joint": best_joint,
                        "lam_ai": lam_ai, "lam_phys": lam_phys, "lam_fac": lam_fac,
                        "facies_warmup_epochs": facies_warmup_epochs,
                        "cc_facies_thresh": cc_facies_thresh,
                        "dt_ms": dt_ms, "f0_hz": f0_hz, "wavelet_nt": wavelet_nt,
                        "ai_mean": ds.ai_mean, "ai_std": ds.ai_std,
                        "seis_mean": ds.seis_mean, "seis_std": ds.seis_std,
                    },
                    os.path.join(out_dir, "best_joint.pt")
                )
                print(f"  saved: best_joint.pt (best_joint={best_joint:.6f})")
            else:
                bad += 1

        history["best_ai"].append(best_ai)
        history["best_joint"].append(best_joint)

        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # early stopping only in joint stage
        if facies_on and (bad >= patience):
            print(f"Early stopping at epoch {epoch}, best_joint={best_joint:.6f}")
            break

    print("Done.")
    print("best_ai   =", best_ai)
    print("best_joint=", best_joint)


if __name__ == "__main__":
    main()
