"""Tests for the turtle geometry interpreter."""

import math

import pytest

from kolam_r.turtle.interpreter import (
    TurtleInterpreter,
    TurtleState,
    LineSegment,
    LoopMarker,
)


class TestTurtleState:
    """Tests for TurtleState."""

    def test_default_state(self):
        state = TurtleState()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.heading == 0.0

    def test_copy_independence(self):
        state = TurtleState(1.0, 2.0, 45.0)
        copy = state.copy()
        copy.x = 99.0
        assert state.x == 1.0  # Original unchanged


class TestLineSegment:
    """Tests for LineSegment."""

    def test_length(self):
        seg = LineSegment(0, 0, 3, 4)
        assert abs(seg.length - 5.0) < 1e-9

    def test_midpoint(self):
        seg = LineSegment(0, 0, 4, 6)
        assert seg.midpoint == (2.0, 3.0)


class TestTurtleInterpreter:
    """Tests for turtle interpretation of L-system strings."""

    def setup_method(self):
        self.turtle = TurtleInterpreter()

    def test_single_forward(self):
        """F at heading 0 should produce segment (0,0) -> (1,0)."""
        result = self.turtle.interpret("F", angle=90.0, step_length=1.0)
        assert len(result.segments) == 1
        seg = result.segments[0]
        assert abs(seg.x1 - 0.0) < 1e-9
        assert abs(seg.y1 - 0.0) < 1e-9
        assert abs(seg.x2 - 1.0) < 1e-9
        assert abs(seg.y2 - 0.0) < 1e-9

    def test_forward_turn_forward(self):
        """F+F at 90 degrees should produce an L-shape."""
        result = self.turtle.interpret("F+F", angle=90.0, step_length=1.0)
        assert len(result.segments) == 2
        # First segment: (0,0) -> (1,0)
        # + is turn right (clockwise): heading becomes -90 = 270 degrees
        # Second segment from (1,0) heading 270: (1,0) -> (1,-1)
        seg2 = result.segments[1]
        assert abs(seg2.x2 - 1.0) < 1e-9
        assert abs(seg2.y2 - (-1.0)) < 1e-9

    def test_square_closed(self):
        """F+F+F+F at 90 degrees should return to start (closed square)."""
        result = self.turtle.interpret("F+F+F+F", angle=90.0, step_length=1.0)
        assert len(result.segments) == 4
        # Last segment should end near (0, 0)
        last = result.segments[-1]
        assert abs(last.x2 - 0.0) < 1e-9
        assert abs(last.y2 - 0.0) < 1e-9

    def test_left_turn(self):
        """F-F at 90 degrees: left turn should go upward."""
        result = self.turtle.interpret("F-F", angle=90.0, step_length=1.0)
        assert len(result.segments) == 2
        # - is turn left (counter-clockwise): heading becomes 90 degrees
        seg2 = result.segments[1]
        assert abs(seg2.x2 - 1.0) < 1e-9
        assert abs(seg2.y2 - 1.0) < 1e-9

    def test_push_pop_stack(self):
        """[, ] should save and restore turtle state."""
        result = self.turtle.interpret("F[+F]F", angle=90.0, step_length=1.0)
        assert len(result.segments) == 3
        # Third segment should continue from position (1,0) heading East
        seg3 = result.segments[2]
        assert abs(seg3.x1 - 1.0) < 1e-9
        assert abs(seg3.y1 - 0.0) < 1e-9
        assert abs(seg3.x2 - 2.0) < 1e-9

    def test_move_without_draw(self):
        """f should move without drawing."""
        result = self.turtle.interpret("fF", angle=90.0, step_length=1.0)
        assert len(result.segments) == 1
        # First segment starts at (1, 0) after the move
        seg = result.segments[0]
        assert abs(seg.x1 - 1.0) < 1e-9

    def test_loop_marker(self):
        """L should record a loop marker."""
        result = self.turtle.interpret("FLF", angle=90.0, step_length=1.0)
        assert len(result.loop_markers) == 1
        assert abs(result.loop_markers[0].x - 1.0) < 1e-9

    def test_reverse(self):
        """|  should reverse direction by 180 degrees."""
        result = self.turtle.interpret("F|F", angle=90.0, step_length=1.0)
        assert len(result.segments) == 2
        # After reverse, heading is 180 degrees, so F goes in -x direction
        seg2 = result.segments[1]
        assert abs(seg2.x2 - 0.0) < 1e-9
        assert abs(seg2.y2 - 0.0) < 1e-9

    def test_unknown_symbols_ignored(self):
        """Non-terminal and unknown symbols should be silently ignored."""
        result = self.turtle.interpret("FAXBYF", angle=90.0, step_length=1.0)
        assert len(result.segments) == 2  # Only F produces segments

    def test_empty_string(self):
        """Empty string should produce no segments."""
        result = self.turtle.interpret("", angle=90.0)
        assert len(result.segments) == 0

    def test_bounds_tracking(self):
        """Bounding box should track all visited points."""
        result = self.turtle.interpret("F-F", angle=90.0, step_length=2.0)
        assert result.min_x <= 0.0
        assert result.max_x >= 2.0
        assert result.max_y >= 2.0

    def test_step_length(self):
        """Custom step length should scale segment lengths."""
        result = self.turtle.interpret("F", angle=90.0, step_length=5.0)
        seg = result.segments[0]
        assert abs(seg.x2 - 5.0) < 1e-9

    def test_parentheses_ignored(self):
        """Parentheses in axiom notation should be ignored."""
        result = self.turtle.interpret("(F)", angle=90.0)
        assert len(result.segments) == 1
