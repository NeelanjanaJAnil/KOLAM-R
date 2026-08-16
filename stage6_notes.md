# KOLAM-R Stage 6 — Topological Validation Notes

## 1. Executive Summary & Dual Representation Specification

Stage 6 established a rigorous topological validation engine to evaluate structural correctness beyond pixel metrics. Because Kolam patterns are governed by Eulerian loop properties and interlaced knotwork, pixel-level metrics (e.g. MSE, SSIM) can score high even when continuous loops are broken or spurious intersections are formed.

$$\text{Image } X \longrightarrow \begin{cases} \text{Medial Thinning } \longrightarrow \text{Junction Graph } G=(V,E) \longrightarrow \beta_0^G, \beta_1^G = |E| - |V| + \beta_0^G \\ \text{GUDHI 2D Cubical Complex } \longrightarrow \text{Persistent Homology } \longrightarrow \beta_0^M, \beta_1^M = \dim(H_1) \end{cases}$$

### Explicit Distinction: Graph-$\beta_1$ vs. Mask-$\beta_1$
> [!IMPORTANT]
> **Graph-$\beta_1$ and Mask-$\beta_1$ are distinct mathematical invariants and are never averaged or conflated:**
> 1. **Graph-$\beta_1$ (Stroke Skeleton Cycle Rank)**: Measures the 1-dimensional homology rank (number of independent cycles / Eulerian loops) in the 1-pixel skeleton graph $G=(V, E)$.
> 2. **Mask-$\beta_1$ (Raster Cubical Complex 1-Homology)**: Measures the number of completely enclosed 2D background holes/cavities within the foreground pixel mask.

---

## 2. Mandatory Gating Unit Tests on Known Geometric Shapes

Before evaluating any Kolam reconstructions, the topological pipeline was strictly verified against known ground-truth shapes (`tests/test_betti_numbers.py`). All four shapes passed with 100% accuracy:

| Test Shape | Description | Expected $\beta_0^G, \beta_1^G$ | Expected $\beta_0^M, \beta_1^M$ | Empirical Result | Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Shape (a)** | Single connected open curve | $(1, 0)$ | $(1, 0)$ | Graph: $(1, 0)$, Mask: $(1, 0)$ | **PASSED** |
| **Shape (b)** | Single closed loop (ring) | $(1, 1)$ | $(1, 1)$ | Graph: $(1, 1)$, Mask: $(1, 1)$ | **PASSED** |
| **Shape (c)** | Two disconnected loops | $(2, 2)$ | $(2, 2)$ | Graph: $(2, 2)$, Mask: $(2, 2)$ | **PASSED** |
| **Shape (d)** | Connected figure-8 (2 cycles) | $(1, 2)$ | $(1, 2)$ | Graph: $(1, 2)$, Mask: $(1, 2)$ | **PASSED** |

---

## 3. Mathematical Disambiguation of Space-Filling Overlap (R02 vs. R05)

In Stage 1 and Stage 3, R02 (Naga / Snake Kolam) and R05 (Hilbert Meander) exhibited high visual overlap under raster SSIM (SSIM 0.2659, 20-25% classification confusion at higher depths). 

Evaluating the topological invariants on the ground-truth synthetic generator proves that **the topological representation cleanly separates these two rule families in the ground truth, providing an unambiguous structural invariant where raster pixel SSIM was ambiguous**:
- **Ground-Truth R02 (Snake Kolam)**: A self-intersecting knotwork pattern with **$\beta_1^G = 5$ independent stroke loops** ($\beta_1^M = 5$ enclosed cavities).
- **Ground-Truth R05 (Hilbert Curve)**: An open, non-intersecting Hamiltonian space-filling meander with **$\beta_1^G = 0$ stroke cycles** ($\beta_1^M = 0$ enclosed cavities).

$$\beta_1(\text{GT-R02}) = 5 \quad \gg \quad \beta_1(\text{GT-R05}) = 0 \implies \text{Topologically Separable in Representation: True}$$

---

## 4. Benchmark Topological Comparison Matrix & Diagnostics

| Benchmark Split | Method | Exact Graph Topo Match (%) ↑ | Mean $\Delta \beta_0^G$ ↓ | Median $\Delta \beta_0^G$ ↓ | Mean $\Delta \beta_1^G$ ↓ | Exact Mask Topo Match (%) ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **In-Distribution (`test_iid`)** | **Grammar Synthesis** | 14.35% | 6.14 | **2.00** | 15.76 | 8.80% |
| | Baseline CNN | 29.17% | 2.78 | 1.00 | 9.03 | 25.00% |
| **Held-Out Depth (`test_heldout_depth`)** | **Grammar Synthesis** | **7.50%** | **1.82** | **0.00** | 17.31 | **0.42%** |
| *(Extrapolation Benchmark)* | Baseline CNN | 5.00% | 3.81 | 1.00 | 17.27 | 0.21% |
| **Gaussian Noise Corruption** | **Grammar Synthesis** | **6.48%** | **13.51** | **6.00** | **23.08** | **5.56%** |
| | Baseline CNN | 0.00% | 27.64 | 18.00 | 32.61 | 0.00% |

### Distribution Analysis of $\Delta \beta_0$ on `test_iid` (Mean 6.14 vs. Median 2.00):
The distribution of connected-component error $\Delta \beta_0$ on `test_iid` reveals that the mean of 6.14 is skewed by a heavy tail of high-symmetry / corrupted outlier samples:
- **Median $\Delta \beta_0$**: **2.00** (25th Percentile: 0.00, 75th Percentile: 8.00).
- **Exact Match ($\Delta \beta_0 = 0$)**: **42.1%** of all samples.
- **Low Error ($\Delta \beta_0 \le 1$)**: **46.3%** of all samples.
- **Catastrophic Outliers ($\Delta \beta_0 \ge 10$)**: **21.3%** of samples (primarily $D_4$ symmetries where 8-fold replication multiplies component count).

### Analysis of the Topology Gap in Paired Depth-Matched Subset (Group a, N=114):
In the paired subset where depth was correctly discovered:
- **Grammar Synthesis**: Exact Graph Topo Match = **21.93%**, Median $\Delta \beta_0 = 0.00$ (Mean 2.51), Mean $\Delta \beta_1 = 7.06$.
- **Baseline CNN**: Exact Graph Topo Match = **46.49%**, Median $\Delta \beta_0 = 0.00$ (Mean 1.66), Mean $\Delta \beta_1 = 5.60$.

**Why does the CNN baseline maintain an edge in exact topology match (46.49% vs. 21.93%)?**
1. **Discrete Rule Lookup vs. Autoregressive Token Generation**: The CNN baseline predicts a discrete `rule_id`, which retrieves the exact pre-registered Stage 1 grammar template from `RULE_REGISTRY` (guaranteeing mathematical loop closure whenever `rule_id` is correct). In contrast, the Vision-to-Grammar model generates 30–50 tokens autoregressively character-by-character; a single inverted turn token (`+` instead of `-`) or displaced stack bracket `[` `]` prevents a loop from closing, altering $\beta_1$.
2. **Symmetry Prediction Multipliers**: In Group (a), **28.1% (32/114) of samples had a symmetry prediction error** (e.g. predicting $C_2$ instead of $C_4$). Because symmetry operations replicate graph components by $|G| \in \{1, 2, 4, 8\}$, an error in symmetry doubles or quadruples both $\beta_0$ and $\beta_1$.

### Balanced Framing on Held-Out Depth Extrapolation:
On held-out recursion depths ($d=3, 4$), exact topological match rates are low for both models (**7.50% for Grammar Synthesis vs. 5.00% for Baseline CNN**, with cycle errors $\Delta \beta_1$ tied at 17.31 vs. 17.27). 
*Key Takeaway*: Unlike raw pixel/geometric coverage where Grammar Synthesis achieved a decisive win (Stage 5), **recovering exact topological loop invariants under recursive scale extrapolation is substantially more challenging**. Recursive expansion at $d=4$ replicates any single local token discrepancy across $2^4=16$ or $4^4=256$ branches, compounding topological cycle differences even when global geometric coverage remains high.

---

## 5. Skeletonization Algorithm & Failure-Case Analysis

### Algorithm Specified:
- **Thinning**: Parallel Zhang-Suen morphological thinning with 8-connectivity preservation and boundary padding.
- **Junction Extraction**: Node clustering across radius $r=2$ and branch tracing with short spur pruning ($\le 3$ px).

### Empirical Failure Cases:
Visual failure cases are documented in [`results/skeleton_failure_cases.png`](file:///C:/Users/lenovo/OneDrive/Desktop/SEM%202/MATERIALS/Kolam/results/skeleton_failure_cases.png):
1. **Dot-Grid Merging**: At higher grid sizes ($g=7, 9$), background pulli dots located close to high-curvature stroke loops can merge into the skeleton under binary thresholding, creating artificial degree-3 T-junctions.
2. **Dense Stroke Fusing**: At $d=4$, the stroke spacing in dense recursive tiles approaches 1-2 pixels. Medial thinning fuses adjacent parallel strokes into singular branches, altering the observed $\beta_1$ cycle count.
3. **Implication for Downstream Stages**: In synthetic evaluation, exact vector stroke graphs provide noise-free ground truth; for raster images, topological invariants remain robust macro-structural indicators but exhibit known degradation when stroke gap $< 2$ px.
