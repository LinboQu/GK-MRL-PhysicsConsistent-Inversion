#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DATA_ROOT>"
  echo "Example: $0 /path/to/data"
  exit 1
fi

DATA_ROOT="$1"
export DATA_ROOT

echo "Using DATA_ROOT=${DATA_ROOT}"

echo "[1/5] Build constraints (constraints.npz)"
python -m src.prepare_constraints

echo "[2/5] Build train/val/test splits"
python -m src.build_splits --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

echo "[3/5] Train baseline model"
python -m src.train_baseline

echo "[4/5] Train physics-consistent model"
python -m src.train_physics

echo "[5/5] Train physics-consistent model (weighted C)"
python -m src.train_physics_weighted

echo "Optional: multi-task facies training"
echo "  python -m src.train_multitask_facies"
