"""Unit tests for the end-to-end forward reconstruction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.reconstruction.pipeline import ReconstructionPipeline, ReconstructionResult
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.schema import BoundingBox, KolamParams
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import LineSegment, TurtleInterpreter


class TestReconstructionPipeline:
    """Tests for forward synthesis pipeline execution."""

    @pytest.fixture
    def sample_target_image(self) -> np.ndarray:
        """Render an authentic R01 depth=1 target image."""
        rule = RULES_BY_ID["R01"]
        interpreter = TurtleInterpreter()
        turtle_result = interpreter.interpret(rule.axiom, angle=45.0)
        sym_segments = apply_symmetry(turtle_result.segments, symmetry="C1")
        bbox = BoundingBox(
            min_x=turtle_result.min_x,
            min_y=turtle_result.min_y,
            max_x=turtle_result.max_x,
            max_y=turtle_result.max_y,
        )
        return render_kolam(
            segments=sym_segments,
            image_size=64,
            motif="M1",
            grid_size=5,
        )

    def test_pipeline_instantiation_without_checkpoints(self):
        """Pipeline can be instantiated without weights (models remain None)."""
        pipeline = ReconstructionPipeline()
        assert pipeline.grammar_model is None
        assert pipeline.baseline_model is None

    def test_pipeline_loading_real_checkpoints(self, sample_target_image):
        """If checkpoints exist on disk, test both forward synthesis paths."""
        grammar_ckpt = Path("checkpoints/best_grammar_model.pt")
        baseline_ckpt = Path("checkpoints/best_val_model.pt")

        if not grammar_ckpt.exists() or not baseline_ckpt.exists():
            pytest.skip("Model checkpoints not found on disk.")

        pipeline = ReconstructionPipeline(
            grammar_model_path=grammar_ckpt,
            baseline_model_path=baseline_ckpt,
        )
        assert pipeline.grammar_model is not None
        assert pipeline.baseline_model is not None

        # Test Grammar Reconstruction
        res_grammar = pipeline.reconstruct_from_grammar(
            sample_target_image,
            motif="M1",
            use_discovered_depth=True,
            compute_fidelity=True,
        )
        assert isinstance(res_grammar, ReconstructionResult)
        assert res_grammar.image.shape == (64, 64)
        assert res_grammar.mask.shape == (64, 64)
        assert res_grammar.mask.dtype == np.uint8
        assert res_grammar.metrics is not None
        assert res_grammar.metrics.ssim >= 0.0

        # Test Baseline CNN Reconstruction
        res_baseline = pipeline.reconstruct_from_baseline_cnn(
            sample_target_image,
            motif="M1",
            compute_fidelity=True,
        )
        assert isinstance(res_baseline, ReconstructionResult)
        assert res_baseline.image.shape == (64, 64)
        assert res_baseline.mask.shape == (64, 64)
        assert res_baseline.metrics is not None
        assert res_baseline.metrics.ssim >= 0.0
