"""Turtle geometry interpreter for L-system strings.

Converts expanded L-system strings into geometric line segments
using a mathematical turtle model. No tkinter dependency — all
computation is pure coordinate geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineSegment:
    """A 2D line segment from (x1, y1) to (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        """Euclidean length of the segment."""
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def midpoint(self) -> tuple[float, float]:
        """Midpoint of the segment."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class LoopMarker:
    """Position where a decorative loop should be drawn."""

    x: float
    y: float
    heading: float  # degrees


@dataclass
class TurtleState:
    """Mutable turtle state: position and heading."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # degrees, 0 = East (positive x), 90 = North (positive y)

    def copy(self) -> TurtleState:
        """Create an independent copy of this state."""
        return TurtleState(self.x, self.y, self.heading)


@dataclass
class TurtleResult:
    """Result of turtle interpretation: segments, loops, and bounds."""

    segments: list[LineSegment] = field(default_factory=list)
    loop_markers: list[LoopMarker] = field(default_factory=list)
    min_x: float = float("inf")
    min_y: float = float("inf")
    max_x: float = float("-inf")
    max_y: float = float("-inf")

    def update_bounds(self, x: float, y: float) -> None:
        """Update bounding box with a new point."""
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    @property
    def width(self) -> float:
        if self.min_x == float("inf"):
            return 0.0
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        if self.min_y == float("inf"):
            return 0.0
        return self.max_y - self.min_y


class TurtleInterpreter:
    """Interprets L-system strings as turtle geometry.

    The turtle starts at the origin facing East (heading=0).
    Symbols are interpreted as follows:
        F  — move forward by step_length, drawing a line
        f  — move forward by step_length, without drawing
        +  — turn right (clockwise) by angle degrees
        -  — turn left (counter-clockwise) by angle degrees
        |  — reverse direction (turn 180 degrees)
        [  — push current state onto stack
        ]  — pop state from stack
        L  — record a loop marker at current position
        (  — ignored (grouping, used in axiom notation)
        )  — ignored (grouping, used in axiom notation)
        All other symbols are silently ignored (non-terminals).
    """

    def interpret(
        self,
        instructions: str,
        angle: float,
        step_length: float = 1.0,
        initial_heading: float = 0.0,
    ) -> TurtleResult:
        """Convert an L-system string to geometric line segments.

        Args:
            instructions: The expanded L-system string.
            angle: Turning angle in degrees for '+' and '-' commands.
            step_length: Distance the turtle moves for 'F' and 'f'.
            initial_heading: Starting heading in degrees (0=East).

        Returns:
            TurtleResult containing line segments, loop markers, and bounds.
        """
        state = TurtleState(x=0.0, y=0.0, heading=initial_heading)
        stack: list[TurtleState] = []
        result = TurtleResult()

        # Record initial position in bounds
        result.update_bounds(state.x, state.y)

        for ch in instructions:
            if ch == "F":
                # Move forward, drawing a line
                rad = math.radians(state.heading)
                nx = state.x + step_length * math.cos(rad)
                ny = state.y + step_length * math.sin(rad)
                result.segments.append(LineSegment(state.x, state.y, nx, ny))
                state.x = nx
                state.y = ny
                result.update_bounds(nx, ny)

            elif ch == "f":
                # Move forward, no drawing
                rad = math.radians(state.heading)
                state.x += step_length * math.cos(rad)
                state.y += step_length * math.sin(rad)
                result.update_bounds(state.x, state.y)

            elif ch == "+":
                # Turn right (clockwise)
                state.heading = (state.heading - angle) % 360.0

            elif ch == "-":
                # Turn left (counter-clockwise)
                state.heading = (state.heading + angle) % 360.0

            elif ch == "|":
                # Reverse direction
                state.heading = (state.heading + 180.0) % 360.0

            elif ch == "[":
                # Push state
                stack.append(state.copy())

            elif ch == "]":
                # Pop state
                if stack:
                    state = stack.pop()

            elif ch == "L":
                # Loop marker at current position
                result.loop_markers.append(
                    LoopMarker(state.x, state.y, state.heading)
                )

            # All other characters (non-terminals, parentheses, etc.) are silently ignored.

        return result
