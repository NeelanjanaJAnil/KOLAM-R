"""Comprehensive benchmark evaluator for Vision-to-Grammar Model.

Evaluates:
- Token-level accuracy & exact sequence match
- Grammar syntactic executability (% valid L-systems)
- Protocol A (Diagnostic Oracle Depth Evaluation)
- Protocol B (HEADLINE: Deployed End-to-End Analysis-by-Synthesis Pipeline)
- Protocol C (Baseline Control: Direct Depth Head)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from kolam_r.dataset.corruption import CorruptionType
from kolam_r.dataset.loader import KolamDataset
from kolam_r.grammar.executor import GrammarExecutor, ParsedGrammar
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.grammar.vocabulary import PAD_IDX
from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel
from kolam_r.schema import VALID_RULES, VALID_SYMMETRIES


class GrammarEvaluator:
    """Benchmark evaluator for Vision-to-Grammar model."""

    def __init__(
        self,
        model: VisionToGrammarModel,
        device: str | None = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = GrammarTokenizer()
        self.executor = GrammarExecutor()
        self.engine = LSystemEngine()

    def evaluate_dataset(
        self,
        dataset: KolamDataset,
        batch_size: int = 32,
        max_safe_depth: int = 4,
    ) -> dict[str, Any]:
        """Run full evaluation on a dataset split."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_samples = len(dataset)
        total_tokens = 0
        correct_tokens = 0
        exact_seq_matches = 0
        syntactically_valid = 0
        symbolic_expansion_matches = 0

        # Protocol A metrics (Oracle depth)
        proto_a_exact_matches = 0
        proto_a_ssims = []

        # Protocol B metrics (Analysis-by-Synthesis headline)
        proto_b_depth_correct = 0
        proto_b_exact_matches = 0
        proto_b_nccs = []

        # Protocol C metrics (Direct depth head)
        proto_c_depth_correct = 0

        # Auxiliary parameter metrics
        sym_correct = 0
        grid_correct = 0
        angle_diffs = []

        idx_to_sym = VALID_SYMMETRIES
        idx_to_grid = [3, 5, 7, 9]

        with torch.no_grad():
            for imgs_raw, targets in loader:
                if isinstance(imgs_raw, np.ndarray):
                    imgs = torch.from_numpy(imgs_raw).float()
                else:
                    imgs = imgs_raw.float()
                imgs = imgs.to(self.device)

                # Generate grammar token sequences
                gen_token_ids, aux = self.model.generate(imgs, max_length=self.model.max_seq_len)

                # Process batch predictions
                for i in range(imgs.size(0)):
                    rule_idx = int(targets["rule_id"][i])
                    gt_rule_id = VALID_RULES[rule_idx]
                    gt_rule_obj = RULES_BY_ID[gt_rule_id]
                    gt_grammar_str = self.tokenizer.rule_to_grammar_string(gt_rule_obj)
                    gt_token_ids = self.tokenizer.encode(gt_grammar_str, max_length=self.model.max_seq_len)

                    pred_ids = gen_token_ids[i].tolist()
                    pred_str = self.tokenizer.decode(pred_ids)

                    # 1. Token Accuracy
                    for p_t, g_t in zip(pred_ids, gt_token_ids):
                        if g_t != PAD_IDX:
                            total_tokens += 1
                            if p_t == g_t:
                                correct_tokens += 1

                    # 2. Exact Sequence Match
                    is_exact_seq = (pred_str.strip() == gt_grammar_str.strip())
                    if is_exact_seq:
                        exact_seq_matches += 1

                    # 3. Syntactic Validity
                    parsed = self.executor.parse_grammar_string(pred_str)
                    if parsed.is_valid:
                        syntactically_valid += 1

                    # 4. Symbolic Expansion Match at true depth
                    true_depth = int(targets["recursion_depth"][i])
                    true_angle = float(targets["angle"][i])
                    true_sym = idx_to_sym[int(targets["symmetry"][i])]
                    true_grid = int(targets["grid_size"][i])
                    true_motif = "M1"  # Structural evaluation

                    if parsed.is_valid:
                        try:
                            recovered_rule = parsed.to_lsystem_rule(default_angle=true_angle)
                            pred_expanded = self.engine.expand_rule(recovered_rule, depth=true_depth)
                            gt_expanded = self.engine.expand_rule(gt_rule_obj, depth=true_depth)
                            if pred_expanded == gt_expanded:
                                symbolic_expansion_matches += 1
                        except Exception:
                            pass

                    # Auxiliary head predictions
                    pred_sym_idx = int(aux["pred_symmetry"][i].cpu())
                    pred_sym = idx_to_sym[pred_sym_idx]
                    if pred_sym == true_sym:
                        sym_correct += 1

                    pred_grid_idx = int(aux["pred_grid_size"][i].cpu())
                    pred_grid = idx_to_grid[pred_grid_idx]
                    if pred_grid == true_grid:
                        grid_correct += 1

                    pred_angle = float(aux["pred_angle"][i].cpu())
                    angle_diffs.append(abs(pred_angle - true_angle))

                    pred_depth_direct = int(aux["pred_depth_direct"][i].cpu())
                    if pred_depth_direct == true_depth:
                        proto_c_depth_correct += 1

                    # Protocol A: Oracle Depth Execution
                    if parsed.is_valid:
                        try:
                            recon_img_a = self.executor.render_grammar(
                                parsed=parsed,
                                depth=true_depth,
                                angle=pred_angle,
                                symmetry=pred_sym,
                                grid_size=pred_grid,
                                motif=true_motif,
                                canvas_size=64,
                            )
                            # SSIM against target image
                            target_canvas = imgs_raw[i, 0].numpy() if isinstance(imgs_raw, torch.Tensor) else imgs_raw[i, 0]
                            ssim_a = float(np.corrcoef(recon_img_a.flatten(), target_canvas.flatten())[0, 1])
                            proto_a_ssims.append(max(ssim_a, 0.0))
                            if is_exact_seq and pred_sym == true_sym and abs(pred_angle - true_angle) <= 5.0 and pred_grid == true_grid:
                                proto_a_exact_matches += 1
                        except Exception:
                            proto_a_ssims.append(0.0)
                    else:
                        proto_a_ssims.append(0.0)

                    # Protocol B: Deployed Analysis-by-Synthesis Depth Search
                    target_canvas = imgs_raw[i, 0].numpy() if isinstance(imgs_raw, torch.Tensor) else imgs_raw[i, 0]
                    rule_max_safe = gt_rule_obj.max_safe_depth
                    discovered_depth, max_ncc = self.executor.search_depth_ncc(
                        parsed=parsed,
                        target_image=target_canvas,
                        angle=pred_angle,
                        symmetry=pred_sym,
                        grid_size=pred_grid,
                        motif=true_motif,
                        max_safe_depth=rule_max_safe,
                    )
                    proto_b_nccs.append(max_ncc)
                    if discovered_depth == true_depth:
                        proto_b_depth_correct += 1

                    if (
                        is_exact_seq
                        and discovered_depth == true_depth
                        and pred_sym == true_sym
                        and abs(pred_angle - true_angle) <= 5.0
                        and pred_grid == true_grid
                    ):
                        proto_b_exact_matches += 1

        angle_mae = float(np.mean(angle_diffs)) if angle_diffs else 0.0

        return {
            "num_samples": total_samples,
            "token_level_accuracy": float(correct_tokens / max(total_tokens, 1)),
            "exact_sequence_match": float(exact_seq_matches / total_samples),
            "syntactic_executability_rate": float(syntactically_valid / total_samples),
            "symbolic_expansion_match_rate": float(symbolic_expansion_matches / total_samples),
            "auxiliary_parameters": {
                "symmetry_accuracy": float(sym_correct / total_samples),
                "grid_size_accuracy": float(grid_correct / total_samples),
                "angle_mae_degrees": angle_mae,
                "angle_accuracy_within_5deg": float(np.mean([d <= 5.0 for d in angle_diffs])),
            },
            "protocol_b_deployed_pipeline": {
                "depth_discovery_accuracy": float(proto_b_depth_correct / total_samples),
                "full_structural_exact_match": float(proto_b_exact_matches / total_samples),
                "mean_reconstruction_ncc": float(np.mean(proto_b_nccs)) if proto_b_nccs else 0.0,
            },
            "protocol_a_oracle_depth_diagnostic": {
                "full_structural_exact_match": float(proto_a_exact_matches / total_samples),
                "mean_reconstruction_correlation": float(np.mean(proto_a_ssims)) if proto_a_ssims else 0.0,
            },
            "protocol_c_direct_head_baseline": {
                "depth_accuracy": float(proto_c_depth_correct / total_samples),
            },
        }

    def evaluate_all(
        self,
        splits_dir: Path | str,
        results_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive evaluation across all splits and corruptions."""
        splits_dir = Path(splits_dir)
        res_dir = Path(results_dir) if results_dir else Path("results")
        res_dir.mkdir(parents=True, exist_ok=True)

        full_results: dict[str, Any] = {
            "model_type": "VisionToGrammarModel",
            "benchmarks": {},
            "corruptions": {},
        }

        # 1. Standard Splits
        for split_name in ["val", "test_iid", "test_heldout_depth"]:
            split_path = splits_dir / f"{split_name}.json"
            if split_path.exists():
                dataset = KolamDataset(split_path)
                metrics = self.evaluate_dataset(dataset)
                full_results["benchmarks"][split_name] = metrics

        # 2. Corrupted Test Suite (on test_iid)
        test_iid_path = splits_dir / "test_iid.json"
        if test_iid_path.exists():
            for ctype in CorruptionType:
                dataset_corrupt = KolamDataset(
                    test_iid_path, corruption=ctype, corruption_severity=1.0, seed=42
                )
                metrics = self.evaluate_dataset(dataset_corrupt)
                full_results["corruptions"][ctype.value] = metrics

        # Save results JSON
        json_path = res_dir / "grammar_evaluation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        return full_results
