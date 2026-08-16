"""Tests for the parameter schema module."""

import pytest
from pydantic import ValidationError

from kolam_r.schema import KolamParams, BoundingBox, KolamMetadata, RULE_DEFAULT_ANGLES


class TestKolamParams:
    """Tests for KolamParams validation."""

    def test_valid_default_params(self):
        """Default parameters should be valid."""
        params = KolamParams()
        assert params.grid_size == 5
        assert params.symmetry == "C1"
        assert params.motif == "M1"
        assert params.production_rule_id == "R01"
        assert params.recursion_depth == 2

    def test_valid_custom_params(self):
        """Custom valid parameters should pass validation."""
        params = KolamParams(
            grid_size=7,
            symmetry="C4",
            angle=90.0,
            motif="M2",
            production_rule_id="R02",
            recursion_depth=3,
        )
        assert params.grid_size == 7
        assert params.symmetry == "C4"

    def test_grid_size_too_small(self):
        """grid_size < 3 should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(grid_size=2)

    def test_grid_size_too_large(self):
        """grid_size > 11 should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(grid_size=12)

    def test_recursion_depth_too_small(self):
        """recursion_depth < 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(recursion_depth=0)

    def test_recursion_depth_too_large(self):
        """recursion_depth > 5 should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(recursion_depth=6)

    def test_invalid_symmetry(self):
        """Invalid symmetry class should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(symmetry="C3")

    def test_invalid_motif(self):
        """Invalid motif should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(motif="M5")

    def test_invalid_rule_id(self):
        """Invalid rule ID should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(production_rule_id="R99")

    def test_invalid_angle_zero(self):
        """Angle of 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(angle=0.0)

    def test_invalid_angle_360(self):
        """Angle of 360 should raise ValidationError."""
        with pytest.raises(ValidationError):
            KolamParams(angle=360.0)

    def test_angle_mismatch_warning(self):
        """Using an angle that doesn't match the rule's default should warn."""
        with pytest.warns(UserWarning, match="does not match"):
            KolamParams(production_rule_id="R01", angle=90.0)

    def test_angle_match_no_warning(self):
        """Using the rule's default angle should not warn."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # R01 default is 45.0
            KolamParams(production_rule_id="R01", angle=45.0)

    def test_frozen(self):
        """Params should be immutable (frozen)."""
        params = KolamParams()
        with pytest.raises(ValidationError):
            params.grid_size = 7


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_width_and_height(self):
        bbox = BoundingBox(min_x=-1.0, min_y=-2.0, max_x=3.0, max_y=4.0)
        assert bbox.width == 4.0
        assert bbox.height == 6.0


class TestRuleDefaultAngles:
    """Tests for rule default angle mapping."""

    def test_all_rules_have_default_angles(self):
        for rule_id in ["R01", "R02", "R03", "R04", "R05", "R06"]:
            assert rule_id in RULE_DEFAULT_ANGLES
            assert RULE_DEFAULT_ANGLES[rule_id] > 0
