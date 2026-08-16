"""Run comprehensive evaluation of trained Vision-to-Grammar model.

Evaluates:
- Protocol A: Diagnostic Oracle Depth Evaluation
- Protocol B: HEADLINE Deployed Analysis-by-Synthesis Pipeline
- Protocol C: Baseline Control Direct Depth Head
- Evaluated on val, test_iid, test_heldout_depth, and corruptions.

Usage:
    python scripts/evaluate_grammar_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.evaluation.grammar_evaluator import GrammarEvaluator
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel


def main() -> None:
    data_dir = project_root / "data"
    splits_dir = data_dir / "splits"
    checkpoint_path = project_root / "checkpoints" / "best_grammar_model.pt"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Run train_grammar_model.py first.")
        sys.exit(1)

    print("=" * 75)
    print("KOLAM-R Stage 4 — Vision-to-Grammar Model Benchmark Evaluation")
    print("=" * 75)

    model = VisionToGrammarModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model weights from {checkpoint_path} (Trained for {checkpoint.get('epoch', '?')} epochs)")

    evaluator = GrammarEvaluator(model=model)
    full_results = evaluator.evaluate_all(splits_dir, results_dir)

    print("\n" + "=" * 75)
    print("GRAMMAR RECOVERY BENCHMARK SUMMARY")
    print("=" * 75)

    benchmarks = full_results.get("benchmarks", {})
    for split_name, metrics in benchmarks.items():
        aux = metrics["auxiliary_parameters"]
        proto_b = metrics["protocol_b_deployed_pipeline"]
        proto_a = metrics["protocol_a_oracle_depth_diagnostic"]
        proto_c = metrics["protocol_c_direct_head_baseline"]

        print(f"\n[{split_name.upper()} SPLIT — {metrics['num_samples']} samples]")
        print(f"  * Token-Level Accuracy:                     {metrics['token_level_accuracy']*100:.2f}%")
        print(f"  * Exact Grammar Sequence Match:             {metrics['exact_sequence_match']*100:.2f}%")
        print(f"  * Grammar Syntactic Executability:          {metrics['syntactic_executability_rate']*100:.2f}%")
        print(f"  * Symbolic Expansion Match Rate:            {metrics['symbolic_expansion_match_rate']*100:.2f}%")
        print("  --- HEADLINE: PROTOCOL B (Deployed Analysis-by-Synthesis Pipeline) ---")
        print(f"  * Discovered Depth Accuracy:                {proto_b['depth_discovery_accuracy']*100:.2f}%")
        print(f"  * Full Structural Exact Match (B):          {proto_b['full_structural_exact_match']*100:.2f}%")
        print(f"  * Mean Reconstruction NCC:                  {proto_b['mean_reconstruction_ncc']:.4f}")
        print("  --- PROTOCOL A (Diagnostic Oracle Depth Evaluation) ---")
        print(f"  * Full Structural Exact Match (A):          {proto_a['full_structural_exact_match']*100:.2f}%")
        print("  --- PROTOCOL C (Baseline Control: Direct Depth Head) ---")
        print(f"  * Direct Head Depth Accuracy:               {proto_c['depth_accuracy']*100:.2f}%")

    print("\n" + "-" * 75)
    print("CORRUPTION ROBUSTNESS BENCHMARK (Evaluated on Test_IID)")
    print("-" * 75)
    clean_match = benchmarks.get("test_iid", {}).get("protocol_b_deployed_pipeline", {}).get("full_structural_exact_match", 0.0) * 100
    print(f"  - Clean (No corruption): {clean_match:.2f}% Protocol B Structural Match")

    for c_name, metrics in full_results.get("corruptions", {}).items():
        c_acc = metrics["protocol_b_deployed_pipeline"]["full_structural_exact_match"] * 100
        drop = clean_match - c_acc
        print(f"  - {c_name:18s}: {c_acc:5.2f}% Struct Match (Delta = -{drop:5.2f}%) | Exact Seq: {metrics['exact_sequence_match']*100:.1f}%")

    print("\n" + "=" * 75)
    print(f"Detailed JSON results saved to: {results_dir / 'grammar_evaluation.json'}")
    print("Stage 4 Benchmark Evaluation Complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
