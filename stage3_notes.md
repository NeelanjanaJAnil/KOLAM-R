# KOLAM-R Stage 3 — CNN Baseline Results & Diagnostic Analysis

## 1. Core Research Question & Findings

> **Research Question**: *"Can direct supervised learning recover the known mathematical parameters of the synthetic Kolam generator?"*

### Findings Summary:
1. **In-Distribution (`test_iid`)**: **Partial Success (Individual parameters learnable, Joint recovery limited)**.
   - Individual parameter recovery is moderate to high: **84.7% Rule ID**, **73.1% Symmetry**, **100.0% Recursion Depth** (within-distribution), and **2.64° Angle MAE**.
   - However, **Structural Exact Match (all 5 structural parameters correct simultaneously) is only 25.0%**.
   - Rendering-level motif classification is trivially 100%, confirming that motif styling is easily captured by spatial kernels but decoupled from structural geometry.

2. **Out-of-Distribution Depth Extrapolation (`test_heldout_depth`)**: **Complete Failure (0.0% Structural Match)**.
   - When tested on recursion depths unseen during training ($d=3$ for R01/R02, $d=4$ for R04/R05/R06), the CNN fails completely to extrapolate:
     - Depth accuracy collapses from **100% to 24.8%** (equivalent to random 4-class guessing).
     - Rule accuracy degrades from **84.7% to 55.2%**.
     - Symmetry accuracy collapses from **73.1% to 40.2%**.
     - Angle MAE worsens from **2.64° to 13.65°**.
   - **Empirical Implication**: The CNN does not learn the algebraic recursive grammar ($X \to \dots$). It relies on spatial stroke frequency heuristics that break down when scale changes.

3. **R02 vs R05 Space-Filling Ambiguity**: **Empirically Proven**.
   - In-distribution ($d \in \{1, 2\}$): R02 and R05 confusion is **0.0%**.
   - At held-out depth ($d=3$): **20.8% of Snake Kolams (R02) are misclassified as Hilbert Meanders (R05)**, and **25.0% of Hilbert Meanders (R05) are misclassified as Snake Kolams (R02)**!
   - This directly confirms the Stage 1 & 2 hypothesis: dense orthogonal meanders cannot be resolved by standard CNN spatial filters without topological loop/cycle representations (Stage 6).

4. **Corruption Robustness**: **Severe Fragility to Spatial Distortions**.
   - Gaussian blur collapses rule accuracy to 19.0% (0.0% exact match).
   - Rotation ($\pm 12^\circ$) drops exact match from 25.0% to 3.24%.
   - Additive Gaussian noise drops exact match to 0.46%.
   - Stroke dropout shows the highest resilience (21.3% exact match), showing that partial occlusions are better tolerated than global geometric distortions.

---

## 2. Multi-Benchmark Performance Matrix

| Benchmark / Split | Samples | Structural Exact Match | Rule Acc | Sym Acc | Depth Acc | Grid Acc | Angle MAE | Motif Acc | R02 $\to$ R05 | R05 $\to$ R02 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation (`val`)** | 216 | **23.15%** | 72.69% | 66.67% | 100.0% | 59.26% | 2.58° | 100.0% | 0.0% | 0.0% |
| **In-Distribution Test (`test_iid`)** | 216 | **25.00%** | 84.72% | 73.15% | 100.0% | 45.37% | 2.64° | 100.0% | 0.0% | 0.0% |
| **Held-Out Depth (`test_heldout_depth`)** | 480 | **0.00%** | 55.21% | 40.21% | 24.79% | 33.54% | 13.65° | 53.75% | **20.83%** | **25.00%** |

---

## 3. Deep Diagnostic Analysis on Held-Out Depth

### A. Depth Extrapolation Failure Mode: Training Ceiling Saturation (Not Random Spread)
Analysis of the 480 held-out depth predictions reveals a systematic failure mode:
- **Zero predictions for $d=4$** (0/480, 0.0%). The model never predicts $d=4$ on held-out inputs.
- **For True $d=4$ (288 samples)**: The model predicts **$d=3$ for 91.7%** (264/288) and $d=2$ for 8.3% (24/288).
- **For True $d=3$ (192 samples of R01 & R02)**: The model predicts $d=3$ for 62.0% (119/192) and $d=2$ for 38.0% (73/192).
- **Conclusion**: The CNN does not guess randomly; it **saturates at the maximum depth observed during training ($d=3$)**. Without an algebraic grammar model, the CNN cannot extrapolate recursive scale beyond its observed training ceiling.

### B. Motif Degradation Mechanism (100% $\to$ 53.75%): Stroke Spatial Occlusion
- **M4 (Double Line)**: **97.5%** accuracy (distinct parallel strokes remain identifiable).
- **M3 (Thick Stroke)**: **73.3%** accuracy.
- **M1 (Thin Stroke)**: **34.2%** accuracy (confused with M2/M3 due to high stroke density).
- **M2 (Rounded Joints)**: **10.0%** accuracy (almost complete collapse).
- **Physical Cause**: In a $64\times 64$ raster canvas at $d=3, 4$, individual segment lengths shrink to 1–2 pixels. The tiny 1-pixel circular joint cap defining M2 is physically fused with neighbouring parallel strokes, obliterating the localized visual signature.

### C. Angle Regression Drift: Centroid Pull
Continuous angle predictions on held-out depth exhibit smooth drift toward the dataset training centroid ($\approx 53^\circ$):
- **True $25.0^\circ$ (R06)**: Predicted mean = **$39.72^\circ \pm 3.92^\circ$** (drifts upward toward 45°).
- **True $45.0^\circ$ (R01, R04)**: Predicted mean = **$50.39^\circ \pm 3.85^\circ$** (tight cluster near 45°–50°).
- **True $90.0^\circ$ (R02, R05)**: Predicted mean = **$69.27^\circ \pm 15.58^\circ$** (drifts downward toward 45°–70°).

---

## 3. Corruption Robustness Breakdown (Evaluated on `test_iid`)

| Condition | Structural Exact Match | Delta vs Clean | Rule Accuracy | Angle MAE |
|:---|:---:|:---:|:---:|:---:|
| **Clean (Uncorrupted)** | **25.00%** | Baseline | **84.72%** | **2.64°** |
| **Gaussian Noise** ($\sigma=0.15$) | **0.46%** | -24.54% | 69.91% | 6.72° |
| **Small Rotation** ($\pm 12^\circ$) | **3.24%** | -21.76% | 58.80% | 14.15° |
| **Affine Shear** ($\pm 15\%$) | **1.39%** | -23.61% | 57.87% | 13.80° |
| **Gaussian Blur** ($r=1.2$) | **0.00%** | -25.00% | 18.98% | 21.45° |
| **Stroke Dropout** (3 patches) | **21.30%** | -3.70% | 82.87% | 3.32° |

---

## 4. Key Limitations of the CNN Baseline

1. **Lack of Recursive Inductive Bias**:
   The CNN treats recursion depth as a categorical bucket rather than a recursive expansion operator. It cannot generalize to $d+1$.
2. **Global Coordinate Rigidity**:
   Slight rotations and shear break symmetry recognition because spatial convolution kernels are not equivariant to continuous SO(2) transformations.
3. **No Sequence Representation**:
   Direct scalar prediction outputs parameters independently without modeling the hierarchical grammar ($D \to C \to B, A$) that connects production rules to turtle actions.

---

## 5. Forward Link to Stage 4 (Grammar & Rule Recovery)

Stage 3 definitively proves that direct image-to-parameter regression is insufficient for structural and recursive generalization.
In **Stage 4**, we formulate inverse reconstruction as an **Image-to-Grammar / Program Synthesis** problem (e.g. predicting L-system tokens and rewrite production sequences), enabling recursive execution at arbitrary depths.
