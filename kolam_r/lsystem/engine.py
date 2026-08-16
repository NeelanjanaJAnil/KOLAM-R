"""L-system string expansion engine.

Performs deterministic, iterative string rewriting according to
production rules. Includes safety guards against string explosion.
"""

from __future__ import annotations

from kolam_r.lsystem.rules import ProductionRule, get_rule, RULE_REGISTRY


DEFAULT_MAX_STRING_LENGTH = 100_000


class LSystemExpansionError(Exception):
    """Raised when L-system expansion exceeds safety limits."""


class LSystemEngine:
    """Deterministic L-system string expansion engine.

    Expands an axiom string through iterative application of production
    rules up to a specified recursion depth. Includes a configurable
    safety guard to prevent memory exhaustion from exponential string growth.

    Attributes:
        max_string_length: Maximum allowed length for the expanded string.
    """

    def __init__(self, max_string_length: int = DEFAULT_MAX_STRING_LENGTH) -> None:
        self.max_string_length = max_string_length

    def expand(
        self,
        axiom: str,
        productions: dict[str, str],
        depth: int,
    ) -> str:
        """Expand an L-system string through `depth` iterations.

        Each iteration applies all production rules in parallel:
        every symbol in the current string is replaced by its
        production (if one exists) or kept as-is.

        Args:
            axiom: The starting string.
            productions: Mapping of symbol -> replacement string.
            depth: Number of expansion iterations (>=0).

        Returns:
            The fully expanded string after `depth` iterations.

        Raises:
            ValueError: If depth is negative.
            LSystemExpansionError: If the expanded string exceeds max_string_length.
        """
        if depth < 0:
            raise ValueError(f"Recursion depth must be non-negative, got {depth}")

        current = axiom
        for iteration in range(depth):
            expanded = []
            for ch in current:
                expanded.append(productions.get(ch, ch))
            current = "".join(expanded)

            if len(current) > self.max_string_length:
                raise LSystemExpansionError(
                    f"Expanded string length ({len(current)}) exceeded "
                    f"maximum ({self.max_string_length}) at iteration "
                    f"{iteration + 1}/{depth}. Consider reducing recursion_depth "
                    f"or increasing max_string_length."
                )

        return current

    def expand_rule(
        self,
        rule_id: str,
        depth: int | None = None,
    ) -> str:
        """Expand a registered production rule by its ID.

        Args:
            rule_id: Rule identifier (e.g. "R01").
            depth: Recursion depth. If None, uses the rule's max_safe_depth.

        Returns:
            The fully expanded L-system string.

        Raises:
            KeyError: If rule_id is not registered.
        """
        if isinstance(rule_id, ProductionRule):
            rule = rule_id
        else:
            rule = get_rule(rule_id)
        if depth is None:
            depth = rule.max_safe_depth
        return self.expand(rule.axiom, rule.productions, depth)

    def estimate_length(self, rule: ProductionRule, depth: int) -> int:
        """Estimate the expanded string length without performing expansion.

        This is a rough upper-bound estimate based on the maximum
        expansion factor of the production rules.

        Args:
            rule: The production rule.
            depth: Recursion depth.

        Returns:
            Estimated string length (upper bound).
        """
        if not rule.productions:
            return len(rule.axiom)

        max_factor = max(len(v) for v in rule.productions.values())
        # Count how many symbols in axiom are expandable
        expandable = sum(1 for ch in rule.axiom if ch in rule.productions)
        non_expandable = len(rule.axiom) - expandable

        # Rough estimate: expandable symbols grow by max_factor each iteration
        estimated = expandable * (max_factor ** depth) + non_expandable
        return estimated
