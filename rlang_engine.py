"""
RLang-inspired grounding layer.

Implements the core RLang concepts (Factors, Propositions, Effects)
as lightweight Python constructs. Designed so that swapping to the
actual rlang package later only requires replacing these classes
with parsed .rlang file outputs.

The key method is to_llm_context() which serializes the NPC's
grounded knowledge into natural language strings for the LLM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from entities import Shape


@dataclass
class RLangState:
    """
    The NPC's grounded knowledge, structured as RLang primitives.

    Factors      = slices of raw state (position, sight grid)
    Propositions = boolean beliefs derived from accumulated observations
    Effects      = causal knowledge (e.g. "I picked up X by moving to it")

    This object represents ONLY what the NPC has observed — never the
    full game state. It is intentionally goal-agnostic: goal tracking
    (what the NPC is pursuing, what it has collected) lives in the brain.
    """

    world_size: int = 15

    # ── Factors (raw state slices) ──────────────────────────────
    npc_pos: tuple[int, int] = (0, 0)

    # ── Accumulated memory (built from observations over time) ──
    observed_cells: set = field(default_factory=set)
    observed_shapes: list[Shape] = field(default_factory=list)

    # e.g. {"red_triangle": [(5,12)], "blue_circle": [(3,7), (8,2)]}
    shape_locations: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # ── Propositions (boolean beliefs) ──────────────────────────

    @property
    def coverage(self) -> float:
        """Fraction of world explored."""
        return len(self.observed_cells) / (self.world_size ** 2)

    @property
    def explored_regions(self) -> dict[str, bool]:
        """Proposition set: which quadrants has the NPC visited?"""
        mid = self.world_size // 2
        regions = {"NW": False, "NE": False, "SW": False, "SE": False}
        for (x, y) in self.observed_cells:
            if x < mid and y < mid:
                regions["NW"] = True
            elif x >= mid and y < mid:
                regions["NE"] = True
            elif x < mid and y >= mid:
                regions["SW"] = True
            else:
                regions["SE"] = True
        return regions

    def seen_label(self, label: str) -> bool:
        """Generic proposition: has the NPC seen a shape with this label?"""
        return label in self.shape_locations

    # ── Observation processing ──────────────────────────────────

    def observe(self, cells: list[tuple[int, int, Shape | None]]):
        """
        Process a batch of visible cells. This is the NPC's sense step.
        Only information within sight range gets recorded.
        """
        for x, y, shape in cells:
            self.observed_cells.add((x, y))
            if shape is not None and shape not in self.observed_shapes:
                self.observed_shapes.append(shape)
                label = shape.label
                self.shape_locations.setdefault(label, [])
                self.shape_locations[label].append((x, y))

    # ── Serialization → LLM context ────────────────────────────

    def to_llm_context(self) -> list[str]:
        """
        Serialize the NPC's RLang-grounded knowledge into a list of
        natural language strings. This is the handoff to the LLM.

        Each string represents one piece of grounded knowledge.
        The LLM team injects these into the system prompt.

        # TODO: Confirm expected format with Marcus (API layer integration).
        # Current format: list of tagged strings e.g. "[FACTOR] I am at (3,7)."
        # May need to change to a single joined string, JSON, or tool-call schema.
        """
        lines: list[str] = []

        # Identity and exploration state
        lines.append(f"[FACTOR] I am at position ({self.npc_pos[0]}, {self.npc_pos[1]}).")
        lines.append(
            f"[FACTOR] I have explored {len(self.observed_cells)}/{self.world_size**2} "
            f"cells ({self.coverage:.0%} of the world)."
        )

        # Explored regions
        regions = self.explored_regions
        explored = [r for r, v in regions.items() if v]
        unexplored = [r for r, v in regions.items() if not v]
        if explored:
            lines.append(f"[PROPOSITION] I have explored the {', '.join(explored)} region(s).")
        if unexplored:
            lines.append(f"[PROPOSITION] I have NOT explored the {', '.join(unexplored)} region(s).")

        # All observed shapes
        for label, positions in self.shape_locations.items():
            display = label.replace("_", " ")
            pos_str = ", ".join(f"({x},{y})" for x, y in positions)
            lines.append(f"[OBSERVATION] I have seen {display}(s) at: {pos_str}.")

        return lines
