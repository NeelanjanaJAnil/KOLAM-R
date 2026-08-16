# KOLAM-R Stage 2 — Dataset & Partitioning Notes

## 1. Overview

Stage 2 generated a canonical, reproducible synthetic dataset of **1,920 samples** (480 unique mathematical structures $\times$ 4 motif rendering styles) with complete ground truth metadata, systematic partition manifests, evaluation-only corruptions, and statistical reporting.

---

## 2. Dataset Structure & Manifests

All data artifacts are stored in `data/`:

```
data/
├── raw/
│   ├── images/                        # 1,920 grayscale PNG images (64x64)
│   ├── images_hires/                  # 1,920 reference PNG images (256x256)
│   └── metadata/                      # 1,920 complete JSON metadata records
├── splits/
│   ├── train.json                     # 1,008 samples (52.5%) — 252 structures
│   ├── val.json                       # 216 samples (11.25%) — 54 structures
│   ├── test_iid.json                  # 216 samples (11.25%) — 54 structures
│   └── test_heldout_depth.json        # 480 samples (25.0%) — 120 structures
└── stats/
    ├── dataset_statistics.json        # Complete JSON statistical export
    ├── distributions.png              # 6-panel distribution charts
    └── corrupted_samples_grid.png     # Clean vs. Corrupted comparison artifact
```

---

## 3. Split Design & Invariants

### Zero Parameter Leakage Invariant (Programmatically Verified)
The partition is performed at the **structural configuration level** (`structure_id = {rule}_d{depth}_{sym}_g{grid}`), guaranteeing that all 4 motif variants of any given structure reside within the exact same partition:
- **`train` (1,008 images / 252 structures)**: Clean standard training distribution across in-distribution recursion depths.
- **`val` (216 images / 54 structures)**: Validation set for checkpoint selection during CNN baseline training.
- **`test_iid` (216 images / 54 structures)**: In-distribution generalization test set on unseen parameter tuples.
- **`test_heldout_depth` (480 images / 120 structures)**: Out-of-distribution recursive depth extrapolation test set.

> [!NOTE]
> **Programmatic Zero-Leakage Guarantee**:
> In `tests/test_splits.py::test_zero_structural_leakage`, mutual disjointness is verified by asserting that the pairwise set intersections of all structural IDs across `train`, `val`, `test_iid`, and `test_heldout_depth` are strictly empty ($\emptyset$). Because all 4 motif variants are bound to their parent `structure_id`, **no individual `(rule, depth, symmetry, grid, motif)` 5-tuple appears in more than one split**.

### Held-Out Recursion Depths

| Rule ID | Rule Name | Total Depths | Train/Val Depths | Held-Out Test Depth | Samples in `test_heldout_depth` |
|:---|:---|:---:|:---:|:---:|:---:|
| **`R01`** | Krishna Anklets | $d \in \{1, 2, 3\}$ | $d \in \{1, 2\}$ | **$d=3$** | 96 (24 structures $\times$ 4 motifs) |
| **`R02`** | Snake Kolam | $d \in \{1, 2, 3\}$ | $d \in \{1, 2\}$ | **$d=3$** | 96 (24 structures $\times$ 4 motifs) |
| **`R03`** | Kolam Tile | $d \in \{1, 2\}$ | $d \in \{1, 2\}$ | *None (kept in IID)* | 0 (avoids single-depth collapse) |
| **`R04`** | Mango Leaf | $d \in \{1, 2, 3, 4\}$ | $d \in \{1, 2, 3\}$ | **$d=4$** | 96 (24 structures $\times$ 4 motifs) |
| **`R05`** | Hilbert Meander | $d \in \{1, 2, 3, 4\}$ | $d \in \{1, 2, 3\}$ | **$d=4$** | 96 (24 structures $\times$ 4 motifs) |
| **`R06`** | Branching Floral | $d \in \{1, 2, 3, 4\}$ | $d \in \{1, 2, 3\}$ | **$d=4$** | 96 (24 structures $\times$ 4 motifs) |

---

## 4. Evaluation-Only Corruption Suite

The following 5 visual corruptions are implemented in `kolam_r/dataset/corruption.py` and are applied **strictly at evaluation time**:
1. **Gaussian Noise**: Zero-mean additive pixel noise ($\sigma = 0.15 \times 255$).
2. **Small Rotation**: In-plane rotation uniformly sampled in $[-12^\circ, +12^\circ]$.
3. **Affine Distortion**: Shear ($\pm 15\%$) and scale perturbations ($\pm 10\%$).
4. **Gaussian Blur**: Spatial Gaussian filtering ($\text{radius} = 1.2$).
5. **Stroke Dropout**: Cutout patch masking (3 random patches of size $6\dots 14\text{px}$) over active stroke lines.

*Visual verification*: Rendered to [`data/stats/corrupted_samples_grid.png`](file:///C:/Users/lenovo/OneDrive/Desktop/SEM%202/MATERIALS/Kolam/data/stats/corrupted_samples_grid.png).

---

## 5. Quantitative R02 vs R05 Overlap Analysis

Across all **288 shared parameter tuples** $(g, s, d, m)$ between R02 (Snake Kolam) and R05 (Hilbert Meander):

| Metric | Overall Mean | Depth 1 ($d=1$) | Depth 2 ($d=2$) | Depth 3 ($d=3$) |
|:---|:---:|:---:|:---:|:---:|
| **Mean SSIM** | **0.2167** | 0.2458 | 0.1383 | **0.2659** |
| **Mean MSE** | **17,744.5** | 10,881.6 | 20,217.6 | **22,134.5** |
| **Mean NCC** | **0.2767** | 0.3421 | 0.1837 | **0.3045** |

### Explanation of the Non-Monotonic SSIM Curve ($0.2458 \to 0.1383 \to 0.2659$):
This is a genuine **multi-scale geometric phenomenon**:
1. **At $d=1$ (SSIM = 0.2458)**: Both curves consist of very sparse lines; high SSIM is driven by the shared vast background of empty black pixels.
2. **At $d=2$ (SSIM = 0.1383, minimum)**: The distinctive macro-geometry of each rule manifests—R02 unfolds outer perimeter loops while R05 forms 4 quadrant U-turns. Their stroke trajectories occupy completely non-overlapping spatial paths, maximizing divergence.
3. **At $d=3$ (SSIM = 0.2659, peak)**: Strokes become $4\times$ denser, and the rasterized $64\times 64$ images converge to high-frequency orthogonal grid-filling textures where individual macro-loop distinctions blur into similar space-filling meanders.

### Key Takeaway:
At $d=3$, the Structural Similarity (SSIM = 0.2659) and Normalized Cross-Correlation (NCC = 0.3045) confirm substantial pixel-level overlap in orthogonal grid textures between R02 and R05. This establishes a clear quantitative benchmark: Stage 3 CNN baselines are expected to experience higher confusion on R02/R05, setting up the empirical necessity for Stage 6 topological cycle rank validation.
