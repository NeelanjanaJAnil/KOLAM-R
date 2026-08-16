# KOLAM-R Stage 1 — Design Notes

## Overview

Stage 1 implements a deterministic parametric generator for a restricted
family of dotted Kolam patterns using constrained L-systems with
turtle-geometry interpretation.

## Architecture

### Pipeline

```
KolamParams (schema.py)
    ↓
ProductionRule lookup (rules.py)
    ↓
L-system expansion (engine.py)
    ↓
Turtle interpretation → line segments (interpreter.py)
    ↓
Symmetry transform (transforms.py)
    ↓
Image rendering + dot grid overlay (image_renderer.py)
    ↓
KolamResult + KolamMetadata
```

### Lattice Geometry

Square orthogonal lattice (Ner Pulli) with dots placed at integer grid
points. This is the confirmed starting geometry. Alternative lattice
types (e.g. staggered/hexagonal/diamond) are deferred to future stages.

## Production Rules

6 curated rules sourced from published Kolam L-system research:

| ID  | Name             | Angle | Source                                              |
|-----|------------------|-------|-----------------------------------------------------|
| R01 | Krishna Anklets  | 45°   | Prusinkiewicz & Hanan 1989; Paul Bourke archive     |
| R02 | Snake Kolam      | 90°   | Paul Bourke; traditional Naga Kolam                 |
| R03 | Kolam Tile       | 45°   | Prusinkiewicz & Hanan 1989, ABOP                    |
| R04 | Mango Leaf       | 45°   | Traditional Mavilai Kolam; Paul Bourke              |
| R05 | Hilbert Meander  | 90°   | Hilbert 1891; Prusinkiewicz & Lindenmayer 1990      |
| R06 | Branching Floral | 25°   | Prusinkiewicz & Lindenmayer ABOP (plant model adapted) |

### Source Verification Status

Each rule's rendered output has been analyzed against its cited literature source:

- **R01 (Krishna Anklets, 45°)**: **[VERIFIED]** At depth=1, generates a symmetric 4-cornered diamond loop. At depth=2 and depth=3, recursively evolves into the characteristic 8-lobed woven lotus knots matching Prusinkiewicz & Hanan (1989) and Paul Bourke's Kolam archive. C4 and D4 symmetry transforms yield concentric woven mandalas.
- **R02 (Snake Kolam, 90°)**: **[FLAGGED — VISUAL OVERLAP WITH R05 AT DEPTH ≥ 2]** While structurally initialized with a closed 4-box initiator (`F+XF+F+XF`), its recursive expansion (`X -> XF-F-F+XF+F+XF-F-F+X`) is a Sierpinski square-curve variant that rapidly converges to a dense orthogonal grid meander. At recursion depth $d \ge 2$ (and especially $d=3$), the resulting $64\times 64$ rasterization visually converges to a space-filling orthogonal texture nearly identical to R05 (Hilbert Meander).
  - **Downstream ML Risk**: This visual overlap poses a severe ambiguity risk for pure image-to-parameter classifiers (Stage 3) and direct image-to-grammar models (Stage 4). A convolutional network relying on pixel texture alone will exhibit high confusion between R02 and R05 unless topological graph representations (Stage 6 cycle rank $\beta_1$) or token-level sequence structure are leveraged.
- **R03 (Kolam Tile, 45°)**: **[VERIFIED]** Multi-level hierarchical tile system (A, B, C, D) generating braided diamond Sikku Kolam tiles matching the hierarchical tile grammar in Prusinkiewicz & Hanan (1989).
- **R04 (Mango Leaf, 45°)**: **[VERIFIED]** Generates nested diamond/leaf-shaped lobes with progressive fractal boundary detail at depths 1–4, matching Mavilai Kolam patterns.
- **R05 (Hilbert Meander, 90°)**: **[VERIFIED]** Produces standard space-filling Hilbert meander curves on the square grid across recursion depths 1–4. Confirmed matching mathematical Hilbert curve specification.
- **R06 (Branching Floral, 25°)**: **[VERIFIED — FLAGGED AS NON-TRADITIONAL]** Generates fractal branching floral trees with push/pop state stacking. Confirmed matching plant branching L-systems (ABOP); flagged as a non-looping, non-traditional Kolam control rule designed to test branching recovery and angle estimation.

---

## Structural & Mathematical Analyses

### 1. Mathematical Distinctiveness: R03 (Kolam Tile) vs R04 (Mango Leaf)

While R03 and R04 both generate $45^\circ$-oriented diamond-like lobe structures, they are mathematically and grammatically distinct:

| Property | R03 (Kolam Tile) | R04 (Mango Leaf) |
|:---|:---|:---|
| **Grammar Architecture** | **Hierarchical 4-Level Stratified System** ($D \to C \to B, A \to F$) | **Single-Variable Uniform Substitution** ($X \to XFX--XFX$) |
| **Axiom** | `(-D--D)` (grouped tile composite) | `-X--X` (open boundary pair) |
| **Expansion Mechanics** | Rule $D$ calls $C$, which calls $A$ and $B$, which unfold into complementary asymmetric sub-paths with alternating turn chirality | Single uniform recursive replacement of $X$ at each recursion depth |
| **Geometric Topology** | Multi-loop interlocking Sikku mat tiles with internal crossings | Self-similar fractal perimeter lobes with nested diamond boundaries |
| **Chirality / Symmetry** | Asymmetric sub-tiles ($A \ne B$) combined to form balanced tiles | Strict mirror self-similarity in the expansion string |

**Conclusion on Visual Similarity**: The superficial visual resemblance between R03 and R04 is **expected and mathematically natural**: both rules use $45^\circ$ turning angles and an initiator with double-turns (`--` = $90^\circ$ right corner), aligning their primary axes along the diamond diagonals. However, their underlying algebraic generation mechanism (stratified hierarchical tile replacement vs. direct recursive fractal boundary) is fundamentally different, making them an excellent test pair for whether the model recovers generative grammar rather than visual heuristics.

---

## Dot-Grid Enforcement Analysis

### Direct Question: Is dot-avoidance geometrically enforced anywhere in the pipeline?
### Direct Answer: **NO.**

### Detailed Code Verification:
1. **Turtle Interpreter (`turtle/interpreter.py`, lines 110–155)**:
   - The turtle maintains state `(x, y, heading)` and processes tokens (`F`, `+`, `-`, `|`, `[`, `]`).
   - It performs blind forward steps in continuous $\mathbb{R}^2$ space ($x' = x + d\cos\theta, y' = y + d\sin\theta$).
   - It has **zero awareness of dot coordinates** and enforces no obstacle repulsion, collision detection, or trajectory clearance.

2. **Symmetry Transform (`symmetry/transforms.py`, lines 55–155)**:
   - Evaluates $2\times 2$ Euclidean rotation and reflection matrices directly on segment endpoints.
   - Has **zero interaction with the dot lattice**.

3. **Renderer (`renderer/image_renderer.py`, lines 140–165)**:
   - The bounding box of all segments is computed, an affine scale/center transform is determined, and dots are drawn as static pixel ellipses at regular grid coordinates `dx = -half + i * dot_spacing`, `dy = -half + j * dot_spacing`.
   - Line strokes are drawn directly across the raster canvas with `draw.line()`.
   - **No collision checking or clearance verification is performed.** Strokes are free to intersect, graze, or pass directly over dot coordinates.

### Conclusion: Fundamental Structural Gap & Synthetic-to-Real Domain Gap
The dot grid in Stage 1 is an **independent, decorative background overlay**. There is **no mathematical guarantee** that generated strokes do not intersect or cross dots.

> [!IMPORTANT]
> **Core Structural Domain Gap**: Dot-avoidance (where the continuous loop never intersects, touches, or crosses a dot, but loops around it) is not merely cosmetic — it is a **defining mathematical and structural invariant of authentic Pulli Kolam**. 
> Because the current synthetic generator does not enforce dot clearance, the synthetic dataset **does not model this fundamental constraint of authentic Kolam geometry**. 
> This constitutes an explicit **synthetic-to-real domain gap** that will directly impact Stage 9 (Real Kolam Experiments), as real Kolam photographs will possess genuine topological dot-avoidance that the synthetic training distribution does not enforce.

### Deferral Recommendation & Research Impact:
- **Decision**: Deferred for Stages 2–5 so inverse parameter and grammar recovery can be established against the controlled synthetic ground truth.
- **Topological Validation (Stage 6)**: The stroke skeleton graph $G=(V, E)$ will validate the topology of the stroke network itself (cycle rank $\beta_1$ and component count $\beta_0$), which remains mathematically rigorous for the generated curve families.
- **Stage 9 Implication**: When evaluating real Kolams in Stage 9, structural extraction must account for this domain gap.

---

## Tracked Research & Engineering Backlog

The following items are explicitly tracked and must not be silently dropped:

- [ ] **[CRITICAL FOR STAGE 9] Implement Geometric Dot-Avoidance Constraint**:
  - *Option A*: Implement a discrete 2D array grammar / tile-based Kolam generator (following Siromoney et al.) where gesture tiles (loop, crossing, line, corner) are anchored directly to grid cells around dots.
  - *Option B*: Implement an active repulsive clearance / obstacle-avoidance field in `turtle/interpreter.py` that curves stroke trajectories away from dot coordinates.
  - *Target Milestone*: Address before or as part of the Stage 9 Synthetic-to-Real adaptation.
- [ ] **[STAGE 6 PRE-CHECK] R02 vs R05 Topological Discrimination**:
  - Evaluate whether stroke graph cycle rank $\beta_1$ and vertex degree distributions can cleanly distinguish R02 from R05 despite their $64\times 64$ raster pixel overlap.
- [ ] **[FUTURE SCOPE] Structurally Meaningful Motif Primitives**:
  - Transition motifs from rendering-level stroke styling (M1–M4) to topological gesture primitives (e.g. corner loop vs. sharp corner) if grammar recovery requires motif-level structure.

---

## Scope Decisions

### Motifs Are Not Structurally Meaningful (Scope Decision)

**M1–M4 are rendering-level stroke style variations only.**

They do NOT alter the mathematical generative structure of the Kolam
pattern. They affect only the visual appearance of the rendered image
(line thickness, joint rounding, double-line style). This means:

1. Motifs must NOT be treated as recovered mathematical structure in
   later grammar-recovery experiments (Stages 4+).
2. If a model predicts "M2" instead of "M1", this is a rendering
   classification error, not a mathematical structure recovery error.
3. This scope decision may be revisited in a future stage if structurally
   meaningful motif primitives (e.g. geometric tile variants that alter
   the pattern topology) are needed.

### What This Generator Is

- A controlled synthetic data generator for machine learning research
- Uses L-systems as a mathematical formalism for a restricted Kolam family
- Produces ground-truth parameter annotations for every image

### What This Generator Is NOT

- NOT a claim that all Kolams are L-systems
- NOT a claim that all Kolams are fractals
- NOT a reproduction of the historical artist's construction process
- NOT a general-purpose Kolam/Rangoli generator

## Symmetry Implementation

Symmetry is applied as a post-processing transform on the turtle-generated
line segments. The base L-system pattern is generated once, then replicated
under the group action:

| Class | Order | Generators                          |
|-------|-------|-------------------------------------|
| C1    | 1     | Identity                            |
| C2    | 2     | 180° rotation                       |
| C4    | 4     | 90° rotation                        |
| D1    | 2     | 1 reflection                        |
| D2    | 4     | 180° rotation + 2 reflections       |
| D4    | 8     | 90° rotation + 4 reflections        |

Segments are centered at the origin before transformation and
de-duplicated after application of group elements.

## Known Limitations

1. **64×64 resolution**: Fine stroke details are lost at this resolution,
   especially for high recursion depths. The 256×256 reference images
   are provided for visual inspection but are not the training target.

2. **Dot grid alignment**: The dot grid is rendered independently of the
   L-system path. Dots are placed at lattice points centered on the
   image, but the L-system strokes are not constrained to pass through
   or around specific dots. This is a known simplification.

3. **String explosion**: Some production rules (especially R03 Kolam Tile)
   grow rapidly. The `max_safe_depth` field and the engine's safety guard
   prevent memory issues, but this limits the achievable complexity.

4. **Symmetry interaction with L-system**: Applying symmetry as a
   post-transform may produce overlapping or visually cluttered strokes
   for some rule/symmetry combinations. Not all combinations are
   aesthetically meaningful.

5. **R06 is not a traditional Kolam**: The branching floral rule is
   adapted from plant L-system models and does not produce traditional
   Kolam loop/weave patterns. It is included to test the model's ability
   to handle branching (push/pop) structures and non-standard angles.

6. **R02/R05 Pixel Indistinguishability**: At $d \ge 3$, R02 and R05 converge
   to dense orthogonal grid meanders that are nearly indistinguishable in
   $64\times 64$ pixel space, presenting a challenging discrimination test
   for downstream grammar recovery models.

## Dependencies

```
numpy>=1.24
Pillow>=10.0
matplotlib>=3.7
pydantic>=2.0
PyYAML>=6.0
pytest>=7.0
```

No ML libraries (PyTorch, etc.) or topology libraries (GUDHI, scikit-image)
are required at this stage.
