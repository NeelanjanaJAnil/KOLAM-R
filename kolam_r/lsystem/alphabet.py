"""L-system alphabet definition for KOLAM-R.

Defines the symbols used in L-system strings and their semantic meaning
for turtle-geometry interpretation.
"""

from enum import Enum


class Symbol(str, Enum):
    """L-system alphabet symbols and their turtle-geometry semantics.

    Terminal symbols are directly interpreted by the turtle.
    Non-terminal symbols are expanded by production rules and ignored by the turtle.
    """

    # --- Terminal symbols (turtle commands) ---
    FORWARD_DRAW = "F"       # Move forward by step_length, drawing a line
    FORWARD_MOVE = "f"       # Move forward by step_length, without drawing
    TURN_RIGHT = "+"         # Turn right (clockwise) by angle degrees
    TURN_LEFT = "-"          # Turn left (counter-clockwise) by angle degrees
    PUSH_STATE = "["         # Push current state (position + heading) onto stack
    POP_STATE = "]"          # Pop state from stack
    REVERSE = "|"            # Reverse direction (turn 180 degrees)
    LOOP_MARKER = "L"        # Draw a decorative loop around the nearest dot

    # --- Non-terminal symbols (expanded by production rules, ignored by turtle) ---
    VAR_A = "A"
    VAR_B = "B"
    VAR_C = "C"
    VAR_D = "D"
    VAR_X = "X"
    VAR_Y = "Y"
    VAR_R = "R"  # Used in Hilbert curve as non-terminal


# Symbols that the turtle interprets (moves, draws, or changes state)
TURTLE_COMMANDS = {
    Symbol.FORWARD_DRAW,
    Symbol.FORWARD_MOVE,
    Symbol.TURN_RIGHT,
    Symbol.TURN_LEFT,
    Symbol.PUSH_STATE,
    Symbol.POP_STATE,
    Symbol.REVERSE,
    Symbol.LOOP_MARKER,
}

# Symbols interpreted as "draw forward" variants
DRAW_SYMBOLS = {Symbol.FORWARD_DRAW}

# Symbols interpreted as "move without drawing"
MOVE_SYMBOLS = {Symbol.FORWARD_MOVE}

# Non-terminal symbols (expanded by rules, ignored by turtle)
NON_TERMINALS = {
    Symbol.VAR_A,
    Symbol.VAR_B,
    Symbol.VAR_C,
    Symbol.VAR_D,
    Symbol.VAR_X,
    Symbol.VAR_Y,
    Symbol.VAR_R,
}
