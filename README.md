# GK-MRL-PhysicsConsistent-Inversion
Code for geological knowledge–constrained physics-consistent seismic impedance inversion (Stanford VI-E).
# Geological Knowledge–Constrained Physics-Consistent Seismic Impedance Inversion

This repository provides reproducible code for **geological knowledge–constrained** and **physics-consistent** seismic impedance inversion experiments on the **Stanford VI-E** reservoir dataset.

> **Note**: The Stanford VI-E dataset is **NOT** included in this repository. Please follow the dataset instructions below.

---

## 1. Highlights
- Geological knowledge constraint construction: stratigraphic coordinate, trend descriptors, reliability field, and interval mask.
- 2.5D patch-based preprocessing for seismic volumes and constraint packages.
- (Planned) geology-constrained CNN backbone + multi-scale stratigraphic representation transformer.
- (Planned) physics-consistent multi-task learning with forward modeling consistency.

---

## 2. Repository structure
```text
src/          Core implementations (reusable modules)
notebooks/    Step-by-step Jupyter notebooks for experiments
data/         Local data directory (ignored by git)
results/      Outputs (ignored by git)
```

## 3. End-to-end pipeline (CLI)
The notebooks have been mirrored into a runnable command-line pipeline. Provide the dataset path via `DATA_ROOT`
or pass it to the script directly.

```bash
# Option A: run the bundled pipeline script
./scripts/run_pipeline.sh /path/to/data

# Option B: run steps manually
export DATA_ROOT=/path/to/data
python -m src.prepare_constraints
python -m src.build_splits --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
python -m src.train_baseline
python -m src.train_physics
python -m src.train_physics_weighted

# Optional multitask training
python -m src.train_multitask_facies
```
