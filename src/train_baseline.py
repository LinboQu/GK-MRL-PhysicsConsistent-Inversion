from __future__ import annotations

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.geo_constraints import DataPaths
from src.dataset_vie import StanfordVIEWellPatchDataset
from src.models.baseline_geo_cnn import GeoCNNTraceHead
from src.utils.train_utils import set_seed, AverageMeter, ensure_dir


def masked_mse(y_hat: torch.Tensor, y: torch.Tensor, m_center: torch.Tensor | None = None) -> torch.Tensor:
    """
    y_hat, y: [B,T]
    m_center: [B,T] mask in {0,1}
    """
    if m_center is None:
        return torch.mean((y_hat - y) ** 2)
    diff2 = (y_hat - y) ** 2
    w = m_center
    return (diff2 * w).sum() / (w.sum() + 1e-8)


def main():
    set_seed(2026)

    DATA_ROOT = os.environ.get(
        "DATA_ROOT",
        r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data",
    )
    paths = DataPaths(DATA_ROOT)

    constraints_npz = os.path.join(paths.processed_dir, "constraints.npz")
    split_dir = os.path.join(paths.processed_dir, "splits")
    train_idx = np.load(os.path.join(split_dir, "train_idx.npy"))
    val_idx   = np.load(os.path.join(split_dir, "val_idx.npy"))

    ds = StanfordVIEWellPatchDataset(
        paths=paths,
        constraints_npz=constraints_npz,
        patch_hw=4,
        use_masked_y=True,   # y already masked
        normalize=True,
    )

    train_set = Subset(ds, train_idx.tolist())
    val_set   = Subset(ds, val_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCNNTraceHead(in_channels=7, base=32, t=200).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    out_dir = os.path.join(paths.processed_dir, "checkpoints_baseline")
    ensure_dir(out_dir)

    best_val = 1e9
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, 31):
        # ---- train ----
        model.train()
        meter = AverageMeter()

        for batch in train_loader:
            x = batch["x"].to(device)  # [B,1,H,W,T]
            p = batch["p"].to(device)
            c = batch["c"].to(device)
            m = batch["m"].to(device)
            y = batch["y"].to(device)  # [B,T]
            

            # center mask for extra safety
            H = m.shape[2]; W = m.shape[3]
            mc = m[:, 0, H//2, W//2, :]  # [B,T]

            y_hat = model(x, p, c, m)    # [B,T]
            loss = masked_mse(y_hat, y, mc)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            meter.update(loss.item(), n=x.size(0))

        train_loss = meter.avg

        # ---- val ----
        model.eval()
        meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                p = batch["p"].to(device)
                c = batch["c"].to(device)
                m = batch["m"].to(device)
                y = batch["y"].to(device)

                H = m.shape[2]; W = m.shape[3]
                mc = m[:, 0, H//2, W//2, :]

                y_hat = model(x, p, c, m)
                loss = masked_mse(y_hat, y, mc)
                meter.update(loss.item(), n=x.size(0))

        val_loss = meter.avg
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "best_val": best_val,
                    "seis_mean": ds.seis_mean, "seis_std": ds.seis_std,
                    "ai_mean": ds.ai_mean, "ai_std": ds.ai_std,
                },
                ckpt_path
            )
            print("  saved:", ckpt_path)

        # always save history
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Done. Best val =", best_val)


if __name__ == "__main__":
    main()
