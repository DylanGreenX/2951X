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

    Factors  = slices of raw state (position, sight grid)
    Propositions = boolean beliefs derived from accumulated observations
    Effects  = causal knowledge ("I collected a blue circle by moving to it")

    This object represents ONLY what the NPC has observed — never the
    full game state.
    """

    world_size: int = 15

    # ── Factors (raw state slices) ──────────────────────────────
    # Factor npc_pos := S[0:2]
    npc_pos: tuple[int, int] = (0, 0)

    # ── Accumulated memory (built from observations over time) ──
    observed_cells: set = field(default_factory=set)
    observed_shapes: list[Shape] = field(default_factory=list)

    # Indexed by shape label for fast lookup
    # e.g. {"red_triangle": [(5,12)], "blue_circle": [(3,7), (8,2)]}
    shape_locations: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # NPC's own goal tracking
    blue_circles_collected: int = 0
    known_blue_circle_positions: list[tuple[int, int]] = field(default_factory=list)

    # ── Propositions (boolean beliefs) ──────────────────────────
    # These are recomputed from memory each tick.

    @property
    def coverage(self) -> float:
        """Fraction of world explored."""
        return len(self.observed_cells) / (self.world_size ** 2)

    @property
    def seen_any_red_triangle(self) -> bool:
        """Proposition: seen_red_triangle := 'red_triangle' in shape_locations"""
        return "red_triangle" in self.shape_locations

    @property
    def seen_any_triangle(self) -> bool:
        """Proposition: seen_triangle := any key containing 'triangle'"""
        return any("triangle" in k for k in self.shape_locations)

    @property
    def knows_blue_circle_location(self) -> bool:
        """Proposition: knows_target := 'blue_circle' in shape_locations and not all collected"""
        return len(self.known_blue_circle_positions) > 0

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

    # ── Observation processing ──────────────────────────────────

    def observe(self, cells: list[tuple[int, int, Shape | None]]):
        """
        Process a batch of visible cells. This is the NPC's 'sense' step.
        Only information within sight range gets recorded.
        """
        for x, y, shape in cells:
            self.observed_cells.add((x, y))
            if shape is not None and shape not in self.observed_shapes:
                self.observed_shapes.append(shape)
                label = shape.label
                self.shape_locations.setdefault(label, [])
                self.shape_locations[label].append((x, y))

                # Track uncollected blue circles for goal-seeking
                if label == "blue_circle":
                    self.known_blue_circle_positions.append((x, y))

    def record_collection(self, shape: Shape):
        """NPC collected a blue circle (its own goal)."""
        self.blue_circles_collected += 1
        pos = (shape.x, shape.y)
        if pos in self.known_blue_circle_positions:
            self.known_blue_circle_positions.remove(pos)

    # ── Serialization → LLM context ────────────────────────────

    def to_llm_context(self) -> list[str]:
        """
        Serialize the NPC's RLang-grounded knowledge into a list of
        natural language strings. This is the handoff to the LLM.

        Each string represents one piece of grounded knowledge.
        The LLM team injects these into the system prompt.
        """
        lines: list[str] = []

        # Identity and state
        lines.append(f"[FACTOR] I am at position ({self.npc_pos[0]}, {self.npc_pos[1]}).")
        lines.append(f"[FACTOR] I have explored {len(self.observed_cells)}/{self.world_size**2} cells ({self.coverage:.0%} of the world).")

        # Own goal progress
        lines.append(f"[GOAL] I have collected {self.blue_circles_collected} blue circles.")
        if self.known_blue_circle_positions:
            for pos in self.known_blue_circle_positions:
                lines.append(f"[GOAL] I know there is an uncollected blue circle at ({pos[0]}, {pos[1]}).")

        # Propositions about exploration
        regions = self.explored_regions
        explored = [r for r, v in regions.items() if v]
        unexplored = [r for r, v in regions.items() if not v]
        if explored:
            lines.append(f"[PROPOSITION] I have explored the {', '.join(explored)} region(s).")
        if unexplored:
            lines.append(f"[PROPOSITION] I have NOT explored the {', '.join(unexplored)} region(s).")

        # Incidental observations (what the LLM would use to help the player)
        for label, positions in self.shape_locations.items():
            if label == "blue_circle":
                continue  # already covered under GOAL
            display = label.replace("_", " ")
            pos_str = ", ".join(f"({x},{y})" for x, y in positions)
            lines.append(f"[OBSERVATION] I saw {display}(s) at: {pos_str}.")

        # Key proposition for the player's quest
        if self.seen_any_red_triangle:
            locs = self.shape_locations["red_triangle"]
            pos_str = ", ".join(f"({x},{y})" for x, y in locs)
            lines.append(f"[PROPOSITION] I HAVE seen red triangle(s) at: {pos_str}.")
        elif self.seen_any_triangle:
            tri_labels = [k for k in self.shape_locations if "triangle" in k]
            all_locs = []
            for tl in tri_labels:
                all_locs.extend(self.shape_locations[tl])
            pos_str = ", ".join(f"({x},{y})" for x, y in all_locs)
            lines.append(f"[PROPOSITION] I have seen triangles (but not red ones) near: {pos_str}.")
        else:
            lines.append("[PROPOSITION] I have not seen any triangles yet.")

        return lines
