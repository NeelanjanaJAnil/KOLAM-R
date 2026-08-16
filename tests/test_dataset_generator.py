"""Tests for dataset generator and structural configuration enumeration."""

from pathlib import Path
import tempfile
import pytest

from kolam_r.dataset.generator import (
    DatasetGenerator,
    StructuralConfig,
    enumerate_all_params,
    enumerate_structural_configs,
)
from kolam_r.schema import VALID_MOTIFS, VALID_RULES, VALID_SYMMETRIES


class TestDatasetGenerator:
    """Tests for dataset configuration enumeration and generation."""

    def test_structural_configs_count(self):
        """Must enumerate exactly 480 unique mathematical structures."""
        configs = enumerate_structural_configs()
        assert len(configs) == 480
        # All structure IDs must be unique
        struct_ids = [c.structure_id for c in configs]
        assert len(set(struct_ids)) == 480

    def test_total_params_count(self):
        """Must enumerate exactly 1,920 full parameter specifications (480 x 4 motifs)."""
        params_list = enumerate_all_params()
        assert len(params_list) == 1920

    def test_rule_representation(self):
        """All 6 rules must be represented across valid depths."""
        configs = enumerate_structural_configs()
        rules = set(c.production_rule_id for c in configs)
        assert rules == set(VALID_RULES)

    def test_symmetry_representation(self):
        """All 6 symmetries must be represented uniformly."""
        configs = enumerate_structural_configs()
        syms = [c.symmetry for c in configs]
        for s in VALID_SYMMETRIES:
            assert syms.count(s) == 480 // len(VALID_SYMMETRIES)

    def test_small_batch_generation(self):
        """Test generating a small batch writes valid images and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            gen = DatasetGenerator(out_dir)

            # Test generation of 5 samples
            params_list = enumerate_all_params()[:5]
            for i, p in enumerate(params_list):
                img_id = f"K{i+1:06d}"
                res = gen.generator.generate(p, image_id=img_id)
                assert res.image_64.shape == (64, 64)
                assert res.metadata.image_id == img_id
