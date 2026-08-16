"""Run comprehensive multi-benchmark evaluation of trained CNN baseline.

Evaluates on:
1. val (In-distribution validation)
2. test_iid (In-distribution unseen parameter tuples)
3. test_heldout_depth (Out-of-distribution recursive depth extrapolation)
4. test_corrupted (5 evaluation corruptions on test_iid)

Usage:
    python scripts/evaluate_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.evaluation.evaluator import BaselineEvaluator
from kolam_r.models.cnn_baseline import KolamCNNBaseline


def main() -> None:
    data_dir = project_root / "data"
    splits_dir = data_dir / "splits"
    checkpoint_path = project_root / "checkpoints" / "best_val_model.pt"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Run train_baseline.py first.")
        sys.exit(1)

    print("=" * 75)
    print("KOLAM-R Stage 3 — Comprehensive Benchmark Evaluation")
    print("=" * 75)

    # Load model
    model = KolamCNNBaseline()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model weights from {checkpoint_path} (Trained for {checkpoint.get('epoch', '?')} epochs)")

    evaluator = BaselineEvaluator(model=model)
    full_results = evaluator.evaluate_all(splits_dir, results_dir)

    print("\n" + "=" * 75)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 75)

    benchmarks = full_results.get("benchmarks", {})
    for split_name, metrics in benchmarks.items():
        struct = metrics["structural_parameters"]
        print(f"\n[{split_name.upper()} SPLIT — {metrics['num_samples']} samples]")
        print(f"  * Structural Exact Match (All 5 Params): {metrics['structural_exact_match_accuracy']*100:.2f}%")
        print(f"  * Full Exact Match (Struct + Motif):     {metrics['full_exact_match_accuracy']*100:.2f}%")
        print(f"  * Rule ID Accuracy:                      {struct['rule_id_accuracy']*100:.2f}%")
        print(f"  * Symmetry Accuracy:                     {struct['symmetry_accuracy']*100:.2f}%")
        print(f"  * Recursion Depth Accuracy:              {struct['recursion_depth_accuracy']*100:.2f}%")
        print(f"  * Grid Size Accuracy:                    {struct['grid_size_accuracy']*100:.2f}%")
        print(f"  * Angle MAE:                             {struct['angle_mae_degrees']:.2f}° (within 5°: {struct['angle_accuracy_within_5deg']*100:.2f}%)")
        print(f"  * [Rendering] Motif Accuracy:            {metrics['rendering_parameters']['motif_accuracy']*100:.2f}%")
        diag = metrics["pair_confusion_diagnostics"]
        print(f"  * R02 Confused as R05 Rate:              {diag['r02_confused_as_r05_rate']*100:.2f}%")
        print(f"  * R05 Confused as R02 Rate:              {diag['r05_confused_as_r02_rate']*100:.2f}%")

    print("\n" + "-" * 75)
    print("CORRUPTION ROBUSTNESS BENCHMARK (Evaluated on Test_IID)")
    print("-" * 75)
    clean_struct_acc = benchmarks.get("test_iid", {}).get("structural_exact_match_accuracy", 0.0) * 100
    print(f"  - Clean (No corruption): {clean_struct_acc:.2f}% Structural Match")

    for c_name, metrics in full_results.get("corruptions", {}).items():
        c_acc = metrics["structural_exact_match_accuracy"] * 100
        drop = clean_struct_acc - c_acc
        print(f"  - {c_name:18s}: {c_acc:5.2f}% Structural Match (Delta = -{drop:5.2f}%) | Rule Acc: {metrics['structural_parameters']['rule_id_accuracy']*100:.1f}%")

    print("\n" + "=" * 75)
    print(f"Detailed JSON results saved to: {results_dir / 'baseline_evaluation.json'}")
    print("Stage 3 Benchmark Evaluation Complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
