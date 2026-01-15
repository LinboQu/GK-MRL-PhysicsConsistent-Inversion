from __future__ import annotations

import argparse
import os

from src.geo_constraints import DataPaths, build_constraints, save_constraints_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build geological constraints and save constraints.npz.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "DATA_ROOT",
            r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data",
        ),
        help="Path to data root (or set DATA_ROOT env var).",
    )
    parser.add_argument("--sigma-trend", type=float, default=3.0, help="Trend smoothing sigma.")
    parser.add_argument("--sigma-xy-m", type=float, default=500.0, help="Lateral reliability sigma (meters).")
    parser.add_argument("--sigma-t-ms", type=float, default=15.0, help="Vertical reliability sigma (ms).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DataPaths(args.data_root)

    pack = build_constraints(
        paths,
        sigma_trend=args.sigma_trend,
        sigma_xy_m=args.sigma_xy_m,
        sigma_t_ms=args.sigma_t_ms,
    )
    out_npz = os.path.join(paths.processed_dir, "constraints.npz")
    save_constraints_npz(out_npz, pack)

    print("Saved:", out_npz)
    print("P:", pack["P"].shape, pack["P"].dtype)
    print("C:", pack["C"].shape, pack["C"].dtype)
    print("M:", pack["M"].shape, pack["M"].dtype)


if __name__ == "__main__":
    main()
