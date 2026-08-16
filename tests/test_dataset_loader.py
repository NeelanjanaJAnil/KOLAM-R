"""Tests for dataset loader and target encoding."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from PIL import Image

from kolam_r.dataset.loader import (
    KolamDataset,
    RULE_TO_IDX,
    SYM_TO_IDX,
    MOTIF_TO_IDX,
)
from kolam_r.dataset.splits import DatasetSplitter


@pytest.fixture
def mock_dataset_environment():
    """Create a temporary mock dataset folder with manifests and dummy images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        img_dir = root / "images"
        meta_dir = root / "metadata"
        splits_dir = root / "splits"

        for d in [img_dir, meta_dir, splits_dir]:
            d.mkdir(parents=True, exist_ok=True)

        splitter = DatasetSplitter()
        splitter.save_splits(root, splits_dir)

        # Create dummy image files for the first 10 samples
        for i in range(1, 20):
            img_id = f"K{i:06d}"
            img = Image.new("L", (64, 64), 100)
            img.save(img_dir / f"{img_id}.png")

        yield root


class TestDatasetLoader:
    """Tests for KolamDataset loading and batch processing."""

    def test_load_dataset_train_split(self, mock_dataset_environment):
        """Test loading dataset from a split file."""
        split_file = mock_dataset_environment / "splits" / "train.json"
        dataset = KolamDataset(split_file)

        assert len(dataset) == 1008
        assert dataset.split_name == "train"

        # Test indexing first sample
        img, targets = dataset[0]
        assert isinstance(img, np.ndarray)
        assert img.shape == (1, 64, 64)
        assert img.dtype == np.float32
        assert 0.0 <= img.min() <= img.max() <= 1.0

        # Check target dictionary keys
        assert "rule_id" in targets
        assert "symmetry" in targets
        assert "motif" in targets
        assert "recursion_depth" in targets
        assert "angle" in targets
        assert "grid_size" in targets

        assert 0 <= targets["rule_id"] < len(RULE_TO_IDX)
        assert 0 <= targets["symmetry"] < len(SYM_TO_IDX)
        assert 0 <= targets["motif"] < len(MOTIF_TO_IDX)

    def test_dataset_with_corruption(self, mock_dataset_environment):
        """Test dataset on-the-fly evaluation corruption."""
        split_file = mock_dataset_environment / "splits" / "test_iid.json"
        dataset = KolamDataset(split_file, corruption="gaussian_noise", corruption_severity=1.0)

        img, targets = dataset[0]
        assert img.shape == (1, 64, 64)
        assert img.dtype == np.float32
