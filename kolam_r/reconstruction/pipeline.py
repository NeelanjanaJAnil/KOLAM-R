"""End-to-End Reconstruction Pipeline for KOLAM-R.

Integrates neural sequence synthesis (Stage 4) and CNN parameter prediction (Stage 3)
with the forward Stage 1 L-system execution and rendering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kolam_r.grammar.executor import GrammarExecutor, ParsedGrammar
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import RULE_REGISTRY, RULES_BY_ID
from kolam_r.models.cnn_baseline import MultiTaskCNN
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel
from kolam_r.reconstruction.metrics import (
    ReconstructionMetrics,
    compute_reconstruction_metrics,
)
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.schema import BoundingBox, KolamParams
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import LineSegment, TurtleInterpreter


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of forward synthesis reconstruction."""

    method: str  # "grammar_program_synthesis" or "baseline_cnn_parameters"
    image: np.ndarray  # (64, 64) uint8 [0, 255]
    mask: np.ndarray  # (64, 64) uint8 [0, 255] (clean binary foreground mask for Stage 6 topology)
    segments: list[LineSegment]
    bbox: BoundingBox
    parameters: dict[str, Any]
    grammar_string: str
    metrics: ReconstructionMetrics | None = None


class ReconstructionPipeline:
    """Unified forward synthesis reconstruction engine."""

    def __init__(
        self,
        grammar_model_path: str | Path | None = None,
        baseline_model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = GrammarTokenizer()
        self.executor = GrammarExecutor()
        self.lsystem_engine = LSystemEngine()
        self.interpreter = TurtleInterpreter()

        # Load Grammar Model (Stage 4)
        self.grammar_model: VisionToGrammarModel | None = None
        if grammar_model_path is not None and Path(grammar_model_path).exists():
            checkpoint = torch.load(grammar_model_path, map_location=self.device, weights_only=True)
            self.grammar_model = VisionToGrammarModel(
                vocab_size=len(self.tokenizer.vocab),
                max_seq_len=110,
            )
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.grammar_model.load_state_dict(state_dict)
            self.grammar_model.to(self.device)
            self.grammar_model.eval()

        # Load Baseline CNN Model (Stage 3)
        self.baseline_model: MultiTaskCNN | None = None
        if baseline_model_path is not None and Path(baseline_model_path).exists():
            checkpoint = torch.load(baseline_model_path, map_location=self.device, weights_only=True)
            self.baseline_model = MultiTaskCNN()
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.baseline_model.load_state_dict(state_dict)
            self.baseline_model.to(self.device)
            self.baseline_model.eval()

    def _prepare_tensor(self, image: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        """Convert input image to normalized torch.Tensor and (64, 64) uint8 numpy array."""
        if isinstance(image, torch.Tensor):
            if image.ndim == 2:
                img_t = image.unsqueeze(0).unsqueeze(0).float() / 255.0
                raw_np = image.cpu().numpy().astype(np.uint8)
            elif image.ndim == 3:
                img_t = image.unsqueeze(0).float() / 255.0
                raw_np = image[0].cpu().numpy().astype(np.uint8)
            else:
                img_t = image.float() / 255.0
                raw_np = image[0, 0].cpu().numpy().astype(np.uint8)
        else:
            raw_np = image.astype(np.uint8)
            img_t = torch.tensor(raw_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        return img_t.to(self.device), raw_np

    def reconstruct_from_grammar(
        self,
        image: np.ndarray | torch.Tensor,
        motif: str = "M1",
        use_discovered_depth: bool = True,
        oracle_depth: int | None = None,
        compute_fidelity: bool = True,
        target_segments: list[LineSegment] | None = None,
    ) -> ReconstructionResult:
        """Forward synthesis via Vision-to-Grammar Program Synthesis (Stage 4/5).

        Pipeline:
        1. Model predicts tokenized L-system grammar string & auxiliary geometry.
        2. Bounded Analysis-by-Synthesis depth search determines true recursion depth.
        3. Parsed rule expanded through pure L-system engine.
        4. Turtle geometry executed, transformed by symmetry group, and rendered.
        """
        if self.grammar_model is None:
            raise RuntimeError("Grammar model is not loaded in ReconstructionPipeline.")

        img_t, target_raw = self._prepare_tensor(image)

        with torch.no_grad():
            gen_token_ids, aux_preds = self.grammar_model.generate(img_t, max_length=110)

        # Decode grammar string
        tokens_list = gen_token_ids[0].cpu().tolist()
        grammar_str = self.tokenizer.decode(tokens_list, remove_special_tokens=True)
        parsed = self.executor.parse_grammar_string(grammar_str)

        # Auxiliary predictions
        idx_to_sym = ["C1", "C2", "C4", "D1", "D2", "D4"]
        idx_to_grid = [3, 5, 7, 9]
        pred_sym = idx_to_sym[int(aux_preds["pred_symmetry"][0].cpu())]
        pred_grid = idx_to_grid[int(aux_preds["pred_grid_size"][0].cpu())]
        pred_angle = float(aux_preds["pred_angle"][0].cpu())

        # Discovered Depth search (Protocol B) or Oracle
        if not use_discovered_depth and oracle_depth is not None:
            depth = oracle_depth
        else:
            discovered_depth, _ = self.executor.search_depth_ncc(
                parsed=parsed,
                target_image=target_raw,
                angle=pred_angle,
                symmetry=pred_sym,
                grid_size=pred_grid,
                motif=motif,
                max_safe_depth=4,
            )
            depth = discovered_depth

        # Forward execution to geometry
        segments, bbox = self.executor.execute_grammar_to_segments(
            parsed=parsed,
            depth=depth,
            angle=pred_angle,
            symmetry=pred_sym,
        )

        # Raster render
        recon_img = self.executor.render_grammar(
            parsed=parsed,
            depth=depth,
            angle=pred_angle,
            symmetry=pred_sym,
            grid_size=pred_grid,
            motif=motif,
            canvas_size=64,
        )

        recon_mask = (recon_img > 30).astype(np.uint8) * 255

        metrics = None
        if compute_fidelity:
            metrics = compute_reconstruction_metrics(
                target_img=target_raw,
                recon_img=recon_img,
                target_segments=target_segments,
                recon_segments=segments,
            )

        return ReconstructionResult(
            method="grammar_program_synthesis",
            image=recon_img,
            mask=recon_mask,
            segments=segments,
            bbox=bbox,
            parameters={
                "depth": depth,
                "symmetry": pred_sym,
                "angle": pred_angle,
                "grid_size": pred_grid,
                "motif": motif,
            },
            grammar_string=grammar_str,
            metrics=metrics,
        )

    def reconstruct_from_baseline_cnn(
        self,
        image: np.ndarray | torch.Tensor,
        motif: str = "M1",
        compute_fidelity: bool = True,
        target_segments: list[LineSegment] | None = None,
    ) -> ReconstructionResult:
        """Forward synthesis via Direct Parameter Baseline CNN (Stage 3).

        Pipeline:
        1. Model predicts discrete rule ID, symmetry, recursion depth, grid size, angle.
        2. Stage 1 pre-registered ProductionRule looked up in RULE_REGISTRY.
        3. Expanded in LSystemEngine, turtle-interpreted, symmetry applied, rendered.
        """
        if self.baseline_model is None:
            raise RuntimeError("Baseline CNN model is not loaded in ReconstructionPipeline.")

        img_t, target_raw = self._prepare_tensor(image)

        with torch.no_grad():
            outputs = self.baseline_model(img_t)

        rules = ["R01", "R02", "R03", "R04", "R05", "R06"]
        symmetries = ["C1", "C2", "C4", "D1", "D2", "D4"]
        grid_sizes = [3, 5, 7, 9]

        pred_rule_id = rules[int(outputs["rule_id"].argmax(dim=-1)[0].cpu())]
        pred_sym = symmetries[int(outputs["symmetry"].argmax(dim=-1)[0].cpu())]
        pred_depth = int(outputs["recursion_depth"].argmax(dim=-1)[0].cpu()) + 1  # 1-indexed (1-4)
        pred_grid = grid_sizes[int(outputs["grid_size"].argmax(dim=-1)[0].cpu())]
        pred_angle = float(outputs["angle"][0].cpu()) if outputs["angle"].ndim == 1 else float(outputs["angle"][0, 0].cpu())

        # Lookup rule in Stage 1 registry
        rule = RULES_BY_ID[pred_rule_id]
        expanded_str = self.lsystem_engine.expand_rule(rule, depth=pred_depth)
        turtle_result = self.interpreter.interpret(expanded_str, angle=pred_angle)
        sym_segments = apply_symmetry(turtle_result.segments, symmetry=pred_sym)
        bbox = BoundingBox(
            min_x=turtle_result.min_x,
            min_y=turtle_result.min_y,
            max_x=turtle_result.max_x,
            max_y=turtle_result.max_y,
        )

        recon_img = render_kolam(
            segments=sym_segments,
            image_size=64,
            motif=motif,
            grid_size=pred_grid,
        )
        recon_mask = (recon_img > 30).astype(np.uint8) * 255

        metrics = None
        if compute_fidelity:
            metrics = compute_reconstruction_metrics(
                target_img=target_raw,
                recon_img=recon_img,
                target_segments=target_segments,
                recon_segments=sym_segments,
            )

        return ReconstructionResult(
            method="baseline_cnn_parameters",
            image=recon_img,
            mask=recon_mask,
            segments=sym_segments,
            bbox=bbox,
            parameters={
                "rule_id": pred_rule_id,
                "depth": pred_depth,
                "symmetry": pred_sym,
                "angle": pred_angle,
                "grid_size": pred_grid,
                "motif": motif,
            },
            grammar_string=self.tokenizer.rule_to_grammar_string(rule),
            metrics=metrics,
        )
