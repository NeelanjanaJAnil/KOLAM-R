"""Unit tests for Classical Computer Vision Baseline."""

from __future__ import annotations

import numpy as np
import pytest

from kolam_r.baselines.classical_cv import (
    ClassicalCVBaseline,
    detect_symmetry_autocorrelation,
)
from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import TurtleInterpreter


class TestClassicalCVBaseline:
    """Test classical symmetry detection and template matching."""

    def test_symmetry_detection_c4(self):
        """A C4 symmetric pattern must be detected as C4 or D4."""
        engine = LSystemEngine()
        interp = TurtleInterpreter()
        r01 = RULES_BY_ID["R01"]
        expanded = engine.expand_rule(r01, depth=1)
        res = interp.interpret(expanded, angle=45.0)
        sym_segs = apply_symmetry(res.segments, symmetry="C4")
        img = render_kolam(sym_segs, image_size=64, motif="M1")

        detected = detect_symmetry_autocorrelation(img)
        assert detected in ("C4", "D4"), f"Expected C4 or D4, got {detected}"

    def test_symmetry_detection_c1(self):
        """An asymmetric open curve must be detected as C1 or D1."""
        img = np.zeros((64, 64), dtype=np.uint8)
        img[10:30, 20:25] = 255
        detected = detect_symmetry_autocorrelation(img)
        assert detected in ("C1", "D1"), f"Expected C1 or D1, got {detected}"

    def test_predict_and_reconstruct(self):
        """Baseline must return valid ReconstructionResult."""
        baseline = ClassicalCVBaseline()
        target = np.zeros((64, 64), dtype=np.uint8)
        target[20:44, 20:44] = 255

        result = baseline.predict_and_reconstruct(target, motif="M1")
        assert result.image.shape == (64, 64)
        assert result.mask.shape == (64, 64)
        assert result.metrics.psnr >= 0.0
        assert "rule_id" in result.parameters
