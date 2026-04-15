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
import re
from dataclasses import dataclass, field
from entities import Shape
import config

_LOCATION_TO_COORDS = {v: k for k, v in config.NATURAL_LOCATIONS.items()}


def get_natural_object_name(label: str) -> str:
    if label in config.NATURAL_OBJECTS:
        return config.NATURAL_OBJECTS[label]
    if '_' in label:
        color, shape = label.split('_', 1)
        return f"{config.NATURAL_COLORS.get(color, color)} {config.NATURAL_SHAPES.get(shape, shape)}"
    return label.replace('_', ' ')


def get_natural_location_name(x: int, y: int) -> str:
    return config.NATURAL_LOCATIONS.get((x, y), f"coordinates ({x}, {y})")


def extract_coordinates_from_text(text: str) -> list:
    """Extract coordinates from both explicit format and natural location names."""
    coords = []
    for x, y in re.findall(r'\((\d+),\s*(\d+)\)', text):
        coords.append((int(x), int(y)))
    for location, coord in _LOCATION_TO_COORDS.items():
        if location.lower() in text.lower():
            coords.append(coord)
    return coords


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

    # Tick when each shape location was first recorded — drives memory decay.
    # Keyed by (x, y) since shape_locations already uses coord as identity.
    shape_first_tick: dict[tuple[int, int], int] = field(default_factory=dict)

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

    def observe(
        self,
        cells: list[tuple[int, int, Shape | None]],
        current_step: int | None = None,
    ):
        """
        Process a batch of visible cells. This is the NPC's sense step.
        Only information within sight range gets recorded.

        Modality hooks (applied when the corresponding config flag is set):
          - NPC_SELECTIVE_ATTENTION: skip shapes whose color/shape_type does
            not match the configured goal attribute. Cells are still marked
            visited so exploration coverage is unaffected.
          - NPC_MEMORY_DECAY_TICKS: requires current_step; shapes first seen
            more than N ticks before current_step are culled after recording.
        """
        attention = getattr(config, "NPC_SELECTIVE_ATTENTION", None)
        goal_color = getattr(config, "NPC_GOAL_COLOR", None)
        goal_shape = getattr(config, "NPC_GOAL_SHAPE", None)

        for x, y, shape in cells:
            self.observed_cells.add((x, y))
            if shape is None or shape in self.observed_shapes:
                continue
            if attention == "color" and shape.color != goal_color:
                continue
            if attention == "shape" and shape.shape_type != goal_shape:
                continue
            self.observed_shapes.append(shape)
            self.shape_locations.setdefault(shape.label, []).append((x, y))
            if current_step is not None:
                self.shape_first_tick[(x, y)] = current_step

        decay = getattr(config, "NPC_MEMORY_DECAY_TICKS", None)
        if decay is not None and current_step is not None:
            self._apply_decay(current_step, decay)

    def _apply_decay(self, current_step: int, decay_ticks: int) -> None:
        """Drop shapes first seen more than decay_ticks ago."""
        cutoff = current_step - decay_ticks
        stale = {loc for loc, t in self.shape_first_tick.items() if t <= cutoff}
        if not stale:
            return
        self.observed_shapes = [
            s for s in self.observed_shapes if (s.x, s.y) not in stale
        ]
        self.shape_locations = {}
        for s in self.observed_shapes:
            self.shape_locations.setdefault(s.label, []).append((s.x, s.y))
        for loc in stale:
            self.shape_first_tick.pop(loc, None)

    # ── Serialization → LLM context ────────────────────────────

    def to_llm_context(self) -> list[str]:
        """Serialize NPC knowledge into natural Skyrim vocabulary for LLM prompts."""
        lines: list[str] = []

        current_location = get_natural_location_name(self.npc_pos[0], self.npc_pos[1])
        lines.append(f"I am currently {current_location}.")
        lines.append(f"I've explored {self.coverage:.0%} of this region during my travels.")

        regions = self.explored_regions
        explored = [r for r, v in regions.items() if v]
        unexplored = [r for r, v in regions.items() if not v]
        if explored:
            lines.append(f"I have traveled through the {', '.join(explored)} area(s).")
        if unexplored:
            lines.append(f"I have not yet ventured into the {', '.join(unexplored)} region(s).")

        for label, positions in self.shape_locations.items():
            natural_name = get_natural_object_name(label)
            natural_locations = [get_natural_location_name(x, y) for x, y in positions]

            if len(positions) == 1:
                lines.append(f"I found a {natural_name} {natural_locations[0]}.")
            else:
                location_list = ", ".join(natural_locations)
                lines.append(f"I've seen {len(positions)} {natural_name}s at: {location_list}.")

        return lines
