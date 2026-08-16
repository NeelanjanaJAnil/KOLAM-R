"""Tests for dataset partitioning, zero parameter leakage, and held-out depth isolation."""

from pathlib import Path
import tempfile
import json
import pytest

from kolam_r.dataset.splits import (
    DatasetSplitter,
    HELDOUT_DEPTH_RULES,
)
from kolam_r.dataset.generator import (
    enumerate_all_params,
    enumerate_structural_configs,
)


class TestDatasetSplits:
    """Tests for dataset partitioning and split invariants."""

    def test_heldout_depth_isolation(self):
        """Held-out recursion depths must strictly belong to test_heldout_depth."""
        splitter = DatasetSplitter()
        (
            train_structs,
            val_structs,
            test_iid_structs,
            heldout_structs,
        ) = splitter.partition_structures()

        # Check total structure counts: 252 train, 54 val, 54 test_iid, 120 heldout = 480
        assert len(train_structs) == 252
        assert len(val_structs) == 54
        assert len(test_iid_structs) == 54
        assert len(heldout_structs) == 120
        assert (
            len(train_structs)
            + len(val_structs)
            + len(test_iid_structs)
            + len(heldout_structs)
            == 480
        )

        # Verify zero leakage of held-out depths in train, val, or test_iid
        for struct in train_structs + val_structs + test_iid_structs:
            rule = struct.production_rule_id
            depth = struct.recursion_depth
            assert not (
                rule in HELDOUT_DEPTH_RULES and depth == HELDOUT_DEPTH_RULES[rule]
            ), f"Held-out condition ({rule}, d={depth}) leaked into {struct.structure_id}"

        # Verify all held-out structures match the designated rules and depths
        for struct in heldout_structs:
            rule = struct.production_rule_id
            depth = struct.recursion_depth
            assert (
                rule in HELDOUT_DEPTH_RULES and depth == HELDOUT_DEPTH_RULES[rule]
            ), f"Invalid held-out struct: {struct.structure_id}"

    def test_zero_structural_leakage(self):
        """No structure_id should appear in more than one partition."""
        splitter = DatasetSplitter()
        (
            train_structs,
            val_structs,
            test_iid_structs,
            heldout_structs,
        ) = splitter.partition_structures()

        train_ids = set(s.structure_id for s in train_structs)
        val_ids = set(s.structure_id for s in val_structs)
        test_iid_ids = set(s.structure_id for s in test_iid_structs)
        heldout_ids = set(s.structure_id for s in heldout_structs)

        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_iid_ids)) == 0
        assert len(train_ids.intersection(heldout_ids)) == 0
        assert len(val_ids.intersection(test_iid_ids)) == 0
        assert len(val_ids.intersection(heldout_ids)) == 0
        assert len(test_iid_ids.intersection(heldout_ids)) == 0

    def test_save_splits_manifests(self):
        """Test generating and saving manifest JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            splits_dir = dataset_dir / "splits"

            splitter = DatasetSplitter()
            saved = splitter.save_splits(dataset_dir, splits_dir)

            assert len(saved) == 4
            assert (splits_dir / "train.json").exists()
            assert (splits_dir / "val.json").exists()
            assert (splits_dir / "test_iid.json").exists()
            assert (splits_dir / "test_heldout_depth.json").exists()

            with open(splits_dir / "train.json", "r") as f:
                data = json.load(f)
                assert data["total_samples"] == 252 * 4  # 1,008 samples
                assert data["num_unique_structures"] == 252
