# KOLAM-R — Faculty Prototype Demonstration Guide

**Project Title:** Inverse Learning of Generative Grammar for Structured Kolam Patterns  
**Prototype Application:** `app/prototype_app.py`  
**Execution Command:**
```bash
streamlit run app/prototype_app.py
```

---

## 1. What the Prototype Demonstrates

This prototype illustrates the complete neuro-symbolic workflow of **KOLAM-R**, with a strict and transparent distinction between:

### A. Controlled Synthetic Reconstruction (Proof of Generator & Validation Pipeline)
- **Input**: Canonical Kolam generated from a registered grammar ($R01$–$R06$).
- **Pipeline**: Image $\to$ Preprocessing $\to$ Grammar Retrieval $\to$ Parametric Generator $\to$ Vector Reconstruction $\to$ Quantitative & Homological Validation.
- **Outcome**: Exact structural reconstruction ($\text{SSIM} = 1.000$, $\text{IoU} = 1.000$, exact Betti number match $\beta_0, \beta_1$).

### B. Real-World Photograph Inference (Current Research Limitation & Problem Motivation)
- **Input**: Real-world photographed or drawn Kolam (e.g. spiral/floral floor art).
- **Outcome**: Explicit diagnostic failure:  
  `"Current grammar library cannot adequately represent this input pattern."`
- **Scientific Rationale**: Real photographs contain intricate multi-scale loops and non-orthogonal geometries that exceed the 6 registered L-system grammars ($R01$–$R06$).
- **Research Roadmap**: Demonstrates to faculty why Stage 4 (end-to-end neural image-to-grammar sequence generation) is necessary to discover unconstrained, novel grammars beyond a fixed discrete registry.

---

## 2. How to Run the Prototype

From the project root directory:
```bash
streamlit run app/prototype_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 3. Real vs. Prototype Component Audit

| Component | Status in Prototype | Final Research Target (Stages 3–7) |
| :--- | :--- | :--- |
| **Parametric L-System Generator** | **100% Real** (Stage 1 deterministic compiler) | Retained as the downstream vector synthesis engine |
| **Topological Invariants** | **100% Real** (Zhang-Suen skeleton + graph cycle rank) | Combined with GUDHI cubical complex persistence |
| **Similarity Metrics** | **100% Real** (SSIM, IoU, NCC, Mean Pixel Error) | Standardized automated benchmarking suite |
| **Synthetic Grammar Recovery** | **100% Real** (Registered reference matching) | Neural Sequence-to-Sequence Vision Transformer |
| **Real Photo Grammar Discovery** | **Diagnostic Limitation** (Identifies library mismatch) | Autoregressive token generation for novel rules |

---

## 4. Strict Diagnostic Rules Implemented

1. **Diagnostic Failure Display**:
   - If $\text{SSIM} < 0.70$ or $\text{IoU} < 0.60$, the UI displays an explicit warning:
     > *"⚠️ Diagnostic Assessment: Current Grammar Library Cannot Adequately Represent This Input Pattern"*
2. **Topological Homology Status**:
   - Displays **`✓ Topology Preserved (Exact Homology)`** **ONLY** when both $\beta_0^{\text{orig}} = \beta_0^{\text{recon}}$ and $\beta_1^{\text{orig}} = \beta_1^{\text{recon}}$.
   - Otherwise, displays **`✗ Topology Not Preserved`** in high-contrast red.
3. **No Metric Manipulation**:
   - All displayed numbers (SSIM, IoU, NCC, Betti counts) are calculated directly from image arrays without post-hoc scaling or synthetic inflation.

---

## 5. Suggested 2-Minute Faculty Demonstration Script

### Step 1: Research Motivation (30 seconds)
> *"Good morning. Our project, **KOLAM-R**, focuses on neuro-symbolic grammar recovery for traditional Kolam art. Standard vision models output blurry pixels; our system aims to infer the interpretable mathematical program—the formal L-system production rules, turning angle, and symmetry group."*

### Step 2: Demonstrating the Core Pipeline via Synthetic Benchmarks (45 seconds)
1. Click **`⚡ Activate 1-Click Demo Mode (R03 Kolam Tile)`** (or select *Krishna Anklets R01* / *Snake Kolam R02*).
2. Show the **Recovered Mathematical Representation**:
   > *"For our benchmark corpus, the system recovers the exact production rules $A \to \dots$, $B \to \dots$, angle $45^\circ$, and symmetry $D_4$."*
3. Show the **Reconstruction & Homology Validation**:
   > *"The parametric generator compiles the grammar into a high-resolution vector image. Both Structural SSIM and topological Betti numbers ($\beta_0, \beta_1$) match exactly, proving the fidelity of our generative compiler."*

### Step 3: Transparent Handling of Real Photographs (45 seconds)
1. Upload a real-world photo.
2. Point to the diagnostic failure banner and metrics:
   > *"Now, when we test a complex real-world photograph, our diagnostic validator immediately detects the limitation: the current discrete registry of 6 grammars cannot represent these 35 complex topological loops ($\beta_1=35$ vs $\beta_1=0$). The system honestly reports 'Topology Not Preserved'."*
3. Conclude with the research roadmap:
   > *"This diagnostic failure directly motivates the next stage of our research: training our Stage 4 multi-task neural network on our 1,920-sample synthetic dataset to generate open-vocabulary, novel production rules that can express arbitrary real-world Kolam topologies."*

---
