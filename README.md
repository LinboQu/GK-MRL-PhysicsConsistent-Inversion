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
