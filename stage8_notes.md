# Stage 8: Systematic Ablation Studies — Converged Benchmark Report

## 1. Experimental Protocol & Convergence Guarantee
All ablated variants were trained for **25 full epochs** with `CosineAnnealingLR` ($5\times 10^{-4} \to 1\times 10^{-5}$), gradient norm clipping ($1.0$), and batch size 32, ensuring all validation loss trajectories reach visible plateau convergence.

## 2. Raw Numerical Evaluation Matrix

| Ablation Dimension | Configuration | Exact Grammar Match | Per-Token Accuracy (Unpadded) | Syntactic Validity | Protocol B Depth Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Loss Grounding** | Full Multi-Task ($\mathcal{L}_{\text{seq}} + 0.2\mathcal{L}_{\text{aux}}$) | **74.07%** | **72.21%** | **100.0%** | **57.41%** |
| | Sequence-Only ($\mathcal{L}_{\text{seq}}$ alone) | 0.00% | 13.38% | 0.00% | 0.00% |
| **Encoder Backbone** | Custom 4-Layer ConvNet + CoordConv | **74.07%** | **72.21%** | **100.0%** | **57.41%** |
| | ResNet-18 Backbone | 0.00% | 13.25% | 0.46% | 0.46% |
| **Data & Motif Scale** | 100% Data (4 Motifs, $N=1008$) | 74.07% | 72.21% | 100.0% | **57.41%** |
| | 50% Data (4 Motifs, $N=504$, seed=42) | **81.02%** | **61.44%** | 100.0% | 15.74% |
| | Single-Motif M1 (Seen M1 eval) | 72.22% | 57.51% | 100.0% | 11.11% |
| | Single-Motif M1 (Unseen M2–M4 transfer) | 49.38% | 44.11% | 98.15% | 16.67% |
| **Search Metric (Oracle $P^*$)** | NCC / SSIM / IoU / MSE | **100.00%** | — | 100.0% | **100.00%** |
| **Search Metric (Predicted $\hat{P}$)** | NCC / SSIM / IoU / MSE | 74.07% | 72.21% | 100.0% | **57.41%** |

## 3. Artifact Links
- **Curves Artifact**: `results/ablation_curves.png`
- **Raw JSON Artifact**: `results/ablation_evaluation.json`
- **Execution Script**: `scripts/train_converged_ablations.py`
