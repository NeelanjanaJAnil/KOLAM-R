"""Integration tests for the Kolam generator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kolam_r.generator import KolamGenerator, KolamResult
from kolam_r.schema import KolamParams
from kolam_r.metadata import reset_counter


class TestKolamGenerator:
    """Integration tests for the full generation pipeline."""

    def setup_method(self):
        self.gen = KolamGenerator()
        reset_counter(0)

    def test_basic_generation(self):
        """Basic generation should produce valid result."""
        params = KolamParams(
            production_rule_id="R01",
            angle=45.0,
            recursion_depth=2,
            symmetry="C4",
            motif="M1",
            grid_size=5,
        )
        result = self.gen.generate(params)
        assert isinstance(result, KolamResult)
        assert result.image_64.shape == (64, 64)
        assert result.image_256.shape == (256, 256)
        assert result.image_64.dtype == np.uint8
        assert len(result.expanded_string) > 0
        assert len(result.segments_raw) > 0
        assert len(result.segments_final) >= len(result.segments_raw)

    def test_metadata_populated(self):
        """Metadata should be fully populated."""
        params = KolamParams(
            production_rule_id="R02",
            angle=90.0,
            recursion_depth=2,
            symmetry="C1",
        )
        result = self.gen.generate(params)
        meta = result.metadata
        assert meta.image_id.startswith("K")
        assert meta.expanded_string_length == len(result.expanded_string)
        assert meta.num_segments == len(result.segments_raw)
        assert meta.connectivity in ["single_stroke", "multi_stroke", "branching"]
        assert meta.generator_version == "0.1.0"
        assert len(meta.timestamp) > 0

    def test_determinism(self):
        """Same params should produce identical images."""
        params = KolamParams(
            production_rule_id="R01",
            angle=45.0,
            recursion_depth=2,
            symmetry="C1",
        )
        reset_counter(0)
        result1 = self.gen.generate(params, image_id="test_det")
        reset_counter(0)
        result2 = self.gen.generate(params, image_id="test_det")
        np.testing.assert_array_equal(result1.image_64, result2.image_64)

    def test_different_rules_different_images(self):
        """Different rules should produce different images."""
        params1 = KolamParams(
            production_rule_id="R01", angle=45.0, recursion_depth=2, symmetry="C1"
        )
        params2 = KolamParams(
            production_rule_id="R02", angle=90.0, recursion_depth=2, symmetry="C1"
        )
        result1 = self.gen.generate(params1)
        result2 = self.gen.generate(params2)
        assert not np.array_equal(result1.image_64, result2.image_64)

    def test_all_rules_generate(self):
        """All 6 rules should generate without error."""
        from kolam_r.lsystem.rules import list_rules, get_rule
        from kolam_r.schema import RULE_DEFAULT_ANGLES

        for rule_id in list_rules():
            rule = get_rule(rule_id)
            params = KolamParams(
                production_rule_id=rule_id,
                angle=RULE_DEFAULT_ANGLES[rule_id],
                recursion_depth=min(2, rule.max_safe_depth),
                symmetry="C1",
            )
            result = self.gen.generate(params)
            assert result.image_64.shape == (64, 64)
            assert np.any(result.image_64 > 0), f"Rule {rule_id} produced blank image"

    def test_generate_and_save(self):
        """generate_and_save should create files on disk."""
        params = KolamParams(
            production_rule_id="R04",
            angle=45.0,
            recursion_depth=2,
            symmetry="C1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = self.gen.generate_and_save(params, output_dir, image_id="K_TEST")

            assert (output_dir / "images" / "K_TEST.png").exists()
            assert (output_dir / "images_hires" / "K_TEST.png").exists()
            assert (output_dir / "metadata" / "K_TEST.json").exists()

    def test_symmetry_increases_segments(self):
        """Higher symmetry should produce more segments."""
        params_c1 = KolamParams(
            production_rule_id="R01", angle=45.0, recursion_depth=2, symmetry="C1"
        )
        params_c4 = KolamParams(
            production_rule_id="R01", angle=45.0, recursion_depth=2, symmetry="C4"
        )
        r_c1 = self.gen.generate(params_c1)
        r_c4 = self.gen.generate(params_c4)
        assert len(r_c4.segments_final) >= len(r_c1.segments_final)
