"""Clean and rigorous evaluation script for Stage 8 Ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from kolam_r.ablations.model_variants import (
    VisionToGrammarResNet,
    VisionToGrammarSequenceOnly,
)
from kolam_r.dataset.loader import KolamDataset
from kolam_r.grammar.executor import GrammarExecutor
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel
from kolam_r.reconstruction.metrics import (
    compute_iou,
    compute_mse,
    compute_ncc,
    compute_ssim,
)


def evaluate_clean(model: nn.Module, dataset: KolamDataset, device: str = "cpu", is_seq_only: bool = False) -> dict:
    model.eval()
    tokenizer = GrammarTokenizer()
    executor = GrammarExecutor()

    symmetries = ["C1", "C2", "C4", "D1", "D2", "D4"]
    grid_sizes = [3, 5, 7, 9]

    total_samples = len(dataset)
    correct_grammar = 0
    correct_depth = 0
    syntactic_valid = 0

    total_tokens = 0
    matching_tokens = 0

    for idx in range(total_samples):
        rec = dataset.records[idx]
        img_raw = dataset.load_image(idx)
        img_t = torch.from_numpy(img_raw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        true_r = rec["production_rule_id"]
        true_d = int(rec["recursion_depth"])

        tgt_seq = tokenizer.rule_to_grammar_string(RULES_BY_ID[true_r])
        tgt_tokens = tokenizer.encode(tgt_seq, max_length=110, add_special_tokens=True)

        with torch.no_grad():
            gen_tok_tensor, aux_preds = model.generate(img_t)

        gen_tokens = gen_tok_tensor[0].cpu().tolist()
        gen_str = tokenizer.decode(gen_tokens)
        parsed = executor.parse_grammar_string(gen_str)

        # Per-token accuracy (truncated/padded to same length for comparison)
        min_len = min(len(tgt_tokens), len(gen_tokens))
        for t_idx in range(min_len):
            if tgt_tokens[t_idx] == gen_tokens[t_idx]:
                matching_tokens += 1
        total_tokens += max(len(tgt_tokens), len(gen_tokens))

        if parsed.is_valid:
            syntactic_valid += 1

        if gen_str == tgt_seq:
            correct_grammar += 1

        # Protocol B Depth search with executor
        if not is_seq_only and "pred_symmetry" in aux_preds:
            sym_idx = int(aux_preds["pred_symmetry"][0].cpu() if aux_preds["pred_symmetry"].ndim > 0 else aux_preds["pred_symmetry"].cpu())
            grid_idx = int(aux_preds["pred_grid_size"][0].cpu() if aux_preds["pred_grid_size"].ndim > 0 else aux_preds["pred_grid_size"].cpu())
            ang_val = float(aux_preds["pred_angle"][0].cpu() if aux_preds["pred_angle"].ndim > 0 else aux_preds["pred_angle"].cpu())
            sym = symmetries[min(sym_idx, len(symmetries) - 1)]
            grid = grid_sizes[min(grid_idx, len(grid_sizes) - 1)]
            angle = ang_val
        else:
            sym = rec.get("symmetry", "C1")
            grid = int(rec.get("grid_size", 5))
            angle = float(rec.get("angle", 90.0))

        if parsed.is_valid:
            disc_d, _ = executor.search_depth_ncc(
                parsed,
                target_image=img_raw,
                angle=angle,
                symmetry=sym,
                grid_size=grid,
                motif=rec.get("motif", "M1"),
            )
        else:
            disc_d = 1  # Fallback for invalid

        if disc_d == true_d:
            correct_depth += 1

    return {
        "exact_grammar_match_rate": float(correct_grammar / total_samples),
        "per_token_accuracy": float(matching_tokens / total_tokens),
        "syntactic_validity_rate": float(syntactic_valid / total_samples),
        "protocol_b_depth_accuracy": float(correct_depth / total_samples),
    }


def evaluate_oracle_depth_search(dataset: KolamDataset) -> dict[str, float]:
    tokenizer = GrammarTokenizer()
    executor = GrammarExecutor()

    metrics = ["ncc", "ssim", "iou", "mse"]
    metric_acc = {}

    for sm in metrics:
        correct = 0
        for idx in range(len(dataset)):
            rec = dataset.records[idx]
            img_raw = dataset.load_image(idx)
            true_d = int(rec["recursion_depth"])
            r_id = rec["production_rule_id"]

            tgt_seq = tokenizer.rule_to_grammar_string(RULES_BY_ID[r_id])
            parsed = executor.parse_grammar_string(tgt_seq)

            best_d = 1
            best_score = -float("inf") if sm != "mse" else float("inf")

            for d in range(1, 5):
                cand = executor.render_grammar(
                    parsed,
                    depth=d,
                    angle=float(rec.get("angle", 90.0)),
                    symmetry=rec.get("symmetry", "C1"),
                    grid_size=int(rec.get("grid_size", 5)),
                    motif=rec.get("motif", "M1"),
                    canvas_size=64,
                )

                if sm == "ncc":
                    score = compute_ncc(cand, img_raw)
                    if score > best_score:
                        best_score = score
                        best_d = d
                elif sm == "ssim":
                    score = compute_ssim(cand, img_raw)
                    if score > best_score:
                        best_score = score
                        best_d = d
                elif sm == "iou":
                    score = compute_iou(cand, img_raw)
                    if score > best_score:
                        best_score = score
                        best_d = d
                elif sm == "mse":
                    score = compute_mse(cand, img_raw)
                    if score < best_score:
                        best_score = score
                        best_d = d

            if best_d == true_d:
                correct += 1

        metric_acc[sm] = float(correct / len(dataset))

    return metric_acc


def main() -> None:
    print("=" * 70)
    print("RUNNING CLEAN STAGE 8 BENCHMARK AUDIT")
    print("=" * 70)

    device = "cpu"
    splits_root = Path("data/splits")
    test_iid = KolamDataset(splits_root / "test_iid.json")
    test_heldout = KolamDataset(splits_root / "test_heldout_depth.json")

    # 1. Evaluate Oracle Depth Search (All 216 test_iid samples)
    print("Evaluating Oracle Grammar Depth Search across NCC, SSIM, IoU, MSE...")
    oracle_depth_results = evaluate_oracle_depth_search(test_iid)
    print("Oracle Depth Search Results (Exact True Grammar):", oracle_depth_results)

    # 2. Evaluate Full Canonical Model
    m_full = VisionToGrammarModel(d_model=256).to(device)
    ckpt = torch.load("checkpoints/best_grammar_model.pt", map_location="cpu", weights_only=True)
    m_full.load_state_dict(ckpt["model_state_dict"])

    res_full_iid = evaluate_clean(m_full, test_iid, device=device)
    res_full_held = evaluate_clean(m_full, test_heldout, device=device)
    print("\nFull Multi-Task Model on test_iid:", res_full_iid)
    print("Full Multi-Task Model on test_heldout_depth:", res_full_held)

    # Export cleanly to json
    results = {
        "oracle_depth_search_on_test_iid": oracle_depth_results,
        "full_multitask_test_iid": res_full_iid,
        "full_multitask_test_heldout": res_full_held,
    }
    with open("results/clean_ablation_audit.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
