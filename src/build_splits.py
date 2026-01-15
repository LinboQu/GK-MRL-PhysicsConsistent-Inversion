from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

from src.dataset_vie import StanfordVIEWellPatchDataset
from src.geo_constraints import DataPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for VIE dataset.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "DATA_ROOT",
            r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data",
        ),
        help="Path to data root (or set DATA_ROOT env var).",
    )
    parser.add_argument("--patch-hw", type=int, default=4, help="Half window size (patch size = 2*hw+1).")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for split shuffling.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio.")
    return parser.parse_args()


def make_splits(n: int, seed: int, ratios: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int32)
    rng.shuffle(idx)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main() -> None:
    args = parse_args()
    paths = DataPaths(args.data_root)
    constraints_npz = os.path.join(paths.processed_dir, "constraints.npz")

    ds = StanfordVIEWellPatchDataset(
        paths=paths,
        constraints_npz=constraints_npz,
        patch_hw=args.patch_hw,
        use_masked_y=True,
        normalize=True,
    )

    train_idx, val_idx, test_idx = make_splits(
        len(ds),
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )

    split_dir = os.path.join(paths.processed_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(split_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(split_dir, "test_idx.npy"), test_idx)

    print("Saved splits to:", split_dir)
    print("Train:", len(train_idx), "Val:", len(val_idx), "Test:", len(test_idx))


if __name__ == "__main__":
    main()
