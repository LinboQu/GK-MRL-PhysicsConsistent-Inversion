from __future__ import annotations

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.geo_constraints import DataPaths
from src.dataset_vie import StanfordVIEWellPatchDataset
from src.models.baseline_geo_cnn import GeoCNNTraceHead
from src.physics_forward import forward_seismic_from_ai
from src.utils.train_utils import set_seed, AverageMeter, ensure_dir


def masked_mse(y_hat: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    diff2 = (y_hat - y) ** 2
    return (diff2 * m).sum() / (m.sum() + 1e-8)


def main():
    set_seed(2026)

    DATA_ROOT = r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data"
    paths = DataPaths(DATA_ROOT)

    constraints_npz = os.path.join(paths.processed_dir, "constraints.npz")
    split_dir = os.path.join(paths.processed_dir, "splits")
    train_idx = np.load(os.path.join(split_dir, "train_idx.npy"))
    val_idx   = np.load(os.path.join(split_dir, "val_idx.npy"))

    ds = StanfordVIEWellPatchDataset(
        paths=paths,
        constraints_npz=constraints_npz,
        patch_hw=4,
        use_masked_y=True,   # y already masked by interval
        normalize=True,
    )

    train_set = Subset(ds, train_idx.tolist())
    val_set   = Subset(ds, val_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GeoCNNTraceHead(in_channels=7, base=32, t=200).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    out_dir = os.path.join(paths.processed_dir, "checkpoints_physics")
    ensure_dir(out_dir)

    # loss weights
    lam_sup  = 1.0      # AI supervision
    lam_phys = 0.3      # physics consistency (start moderate)

    # forward params
    dt_ms = 1.0
    f0_hz = 40.0
    wavelet_nt = 81

    best_val = 1e9
    history = {"train_loss": [], "val_loss": [], "train_sup": [], "train_phys": [], "val_sup": [], "val_phys": []}

    patience = 8
    bad = 0

    for epoch in range(1, 51):
        # ---- train ----
        model.train()
        meter = AverageMeter()
        meter_sup = AverageMeter()
        meter_phys = AverageMeter()

        for batch in train_loader:
            x = batch["x"].to(device)  # [B,1,H,W,T] seismic patch (normalized)
            p = batch["p"].to(device)
            c = batch["c"].to(device)
            m = batch["m"].to(device)  # [B,1,H,W,T]
            y = batch["y"].to(device)  # [B,T] AI (normalized & masked)

            H = m.shape[2]; W = m.shape[3]
            mc = m[:, 0, H//2, W//2, :]          # [B,T] interval mask at center
            s_obs = x[:, 0, H//2, W//2, :]        # [B,T] observed seismic center trace (normalized)

            y_hat = model(x, p, c, m)             # [B,T] predicted AI

            # supervision loss (AI)
            L_sup = masked_mse(y_hat, y, mc)

            # physics loss: synth seismic from predicted AI
            s_syn = forward_seismic_from_ai(y_hat, dt_ms=dt_ms, f0_hz=f0_hz, wavelet_nt=wavelet_nt)
            L_phys = masked_mse(s_syn, s_obs, mc)

            loss = lam_sup * L_sup + lam_phys * L_phys

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            meter.update(loss.item(), n=x.size(0))
            meter_sup.update(L_sup.item(), n=x.size(0))
            meter_phys.update(L_phys.item(), n=x.size(0))

        train_loss = meter.avg

        # ---- val ----
        model.eval()
        meter = AverageMeter()
        meter_sup = AverageMeter()
        meter_phys = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                p = batch["p"].to(device)
                c = batch["c"].to(device)
                m = batch["m"].to(device)
                y = batch["y"].to(device)

                H = m.shape[2]; W = m.shape[3]
                mc = m[:, 0, H//2, W//2, :]
                s_obs = x[:, 0, H//2, W//2, :]

                y_hat = model(x, p, c, m)

                L_sup = masked_mse(y_hat, y, mc)
                s_syn = forward_seismic_from_ai(y_hat, dt_ms=dt_ms, f0_hz=f0_hz, wavelet_nt=wavelet_nt)
                L_phys = masked_mse(s_syn, s_obs, mc)

                loss = lam_sup * L_sup + lam_phys * L_phys

                meter.update(loss.item(), n=x.size(0))
                meter_sup.update(L_sup.item(), n=x.size(0))
                meter_phys.update(L_phys.item(), n=x.size(0))

        val_loss = meter.avg

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sup"].append(meter_sup.avg)
        history["train_phys"].append(meter_phys.avg)
        history["val_sup"].append(meter_sup.avg)
        history["val_phys"].append(meter_phys.avg)

        print(
            f"Epoch {epoch:02d} | "
            f"train {train_loss:.6f} (sup {history['train_sup'][-1]:.6f}, phys {history['train_phys'][-1]:.6f}) | "
            f"val {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            ckpt_path = os.path.join(out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "best_val": best_val,
                    "lam_sup": lam_sup,
                    "lam_phys": lam_phys,
                    "dt_ms": dt_ms,
                    "f0_hz": f0_hz,
                    "wavelet_nt": wavelet_nt,
                    "seis_mean": ds.seis_mean, "seis_std": ds.seis_std,
                    "ai_mean": ds.ai_mean, "ai_std": ds.ai_std,
                },
                ckpt_path
            )
            print("  saved:", ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}, best={best_val}")
                break

        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Done. Best val =", best_val)


if __name__ == "__main__":
    main()