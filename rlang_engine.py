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
from game_api_interface import get_natural_position_name
import config


# Region phrases get_natural_position_name can produce. Derived from the
# function's codomain so swapping the region naming (e.g. to match a new
# painted map) propagates to extract_regions_from_text + metrics without
# hand-editing this tuple.
def _collect_region_phrases(world_size: int) -> tuple[str, ...]:
    seen: set[str] = set()
    for gx in range(world_size):
        for gy in range(world_size):
            seen.add(get_natural_position_name(gx, gy, world_size))
    # Longest-first so extract_regions_from_text prefers the most specific
    # phrase on any substring overlap.
    return tuple(sorted(seen, key=len, reverse=True))


_REGION_PHRASES: tuple[str, ...] = _collect_region_phrases(config.GRID_SIZE)


def get_natural_object_name(label: str) -> str:
    if label in config.NATURAL_OBJECTS:
        return config.NATURAL_OBJECTS[label]
    if '_' in label:
        color, shape = label.split('_', 1)
        return f"{config.NATURAL_COLORS.get(color, color)} {config.NATURAL_SHAPES.get(shape, shape)}"
    return label.replace('_', ' ')


def extract_coordinates_from_text(text: str) -> list:
    """Extract literal (x, y) coordinates from text."""
    return [(int(x), int(y)) for x, y in re.findall(r'\((\d+),\s*(\d+)\)', text)]


def extract_regions_from_text(text: str) -> set[str]:
    """
    Return the set of region phrases present in text. Longest-first matching so
    the most specific region wins when phrases overlap (e.g. "the deep swamp"
    over "the swamp").
    """
    lower = text.lower()
    found: set[str] = set()
    consumed: list[tuple[int, int]] = []
    for phrase in _REGION_PHRASES:
        idx = 0
        while (idx := lower.find(phrase, idx)) != -1:
            span = (idx, idx + len(phrase))
            if any(s <= span[0] < e or s < span[1] <= e for s, e in consumed):
                idx = span[1]
                continue
            found.add(phrase)
            consumed.append(span)
            idx = span[1]
    return found


def region_of(x: int, y: int, world_size: int) -> str:
    """Canonical region phrase for a cell — the region an answer must match."""
    return get_natural_position_name(x, y, world_size)


def _npc_display_name(npc_id: str) -> str:
    if npc_id.startswith("npc_"):
        suffix = npc_id.split("_")[-1]
        if suffix.isdigit():
            return f"NPC {int(suffix) + 1}"
    return npc_id


def _source_phrase(npc_ids: tuple[str, ...]) -> str:
    if not npc_ids:
        return "another NPC"
    if len(npc_ids) == 1:
        return _npc_display_name(npc_ids[0])
    return "other NPCs"


def _goal_source_phrase(
    subject_npc_id: str,
    source_npc_ids: tuple[str, ...],
) -> str:
    if not source_npc_ids:
        return "another NPC"
    if len(source_npc_ids) == 1 and source_npc_ids[0] == subject_npc_id:
        return _npc_display_name(subject_npc_id)
    return _source_phrase(source_npc_ids)


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
    observed_cell_sources: dict[tuple[int, int], tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    observed_shape_sources: dict[tuple[int, int], tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    known_npc_goals: dict[str, tuple[str, str, tuple[str, ...]]] = field(default_factory=dict)

    # ── Propositions (boolean beliefs) ──────────────────────────

    @property
    def coverage(self) -> float:
        """Fraction of world explored."""
        return len(self.observed_cells) / (self.world_size ** 2)

    @property
    def explored_regions(self) -> dict[str, bool]:
        """Proposition set: which quadrants has the NPC visited?"""
        return self._regions_for_cells(self.observed_cells)

    @property
    def direct_coverage(self) -> float:
        """Fraction of the world personally observed by this NPC."""
        return len(self.direct_observed_cells) / (self.world_size ** 2)

    @property
    def direct_observed_cells(self) -> set[tuple[int, int]]:
        return {
            cell
            for cell, (kind, _) in self.observed_cell_sources.items()
            if kind in {"direct", "perfect"}
        }

    @property
    def learned_observed_cells(self) -> set[tuple[int, int]]:
        return {
            cell
            for cell, (kind, _) in self.observed_cell_sources.items()
            if kind == "interaction"
        }

    @property
    def direct_explored_regions(self) -> dict[str, bool]:
        return self._regions_for_cells(self.direct_observed_cells)

    @property
    def learned_regions(self) -> dict[str, bool]:
        return self._regions_for_cells(self.learned_observed_cells)

    def _regions_for_cells(self, cells: set[tuple[int, int]]) -> dict[str, bool]:
        """Proposition set: which quadrants have the given cells touched?"""
        mid = self.world_size // 2
        regions = {"NW": False, "NE": False, "SW": False, "SE": False}
        for (x, y) in cells:
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
            self.observed_cell_sources[(x, y)] = ("direct", ())
            if shape is None or shape in self.observed_shapes:
                continue
            if attention == "color" and shape.color != goal_color:
                continue
            if attention == "shape" and shape.shape_type != goal_shape:
                continue
            self.observed_shapes.append(shape)
            self.shape_locations.setdefault(shape.label, []).append((x, y))
            self.observed_shape_sources[(x, y)] = ("direct", ())
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
            self.observed_shape_sources.pop(loc, None)

    def replace_memory(
        self,
        *,
        observed_cells: set[tuple[int, int]],
        observed_shapes: list[Shape],
        shape_locations: dict[str, list[tuple[int, int]]],
        shape_first_tick: dict[tuple[int, int], int],
        observed_cell_sources: dict[tuple[int, int], tuple[str, tuple[str, ...]]] | None = None,
        observed_shape_sources: dict[tuple[int, int], tuple[str, tuple[str, ...]]] | None = None,
        known_npc_goals: dict[str, tuple[str, str, tuple[str, ...]]] | None = None,
    ) -> None:
        """Replace embodied memory while preserving the caller-owned npc_pos."""
        self.observed_cells = set(observed_cells)
        self.observed_shapes = list(observed_shapes)
        self.shape_locations = {
            label: list(positions)
            for label, positions in shape_locations.items()
        }
        self.shape_first_tick = dict(shape_first_tick)
        self.observed_cell_sources = {
            cell: (
                kind,
                tuple(sorted(set(npc_ids))),
            )
            for cell, (kind, npc_ids) in (observed_cell_sources or {}).items()
        }
        self.observed_shape_sources = {
            cell: (
                kind,
                tuple(sorted(set(npc_ids))),
            )
            for cell, (kind, npc_ids) in (observed_shape_sources or {}).items()
        }
        for cell in self.observed_cells:
            self.observed_cell_sources.setdefault(cell, ("direct", ()))
        for label, positions in self.shape_locations.items():
            for pos in positions:
                self.observed_shape_sources.setdefault(pos, ("direct", ()))
        self.known_npc_goals = {
            npc_id: (
                goal_label,
                kind,
                tuple(sorted(set(source_npc_ids))),
            )
            for npc_id, (goal_label, kind, source_npc_ids) in (known_npc_goals or {}).items()
            if goal_label
        }

    # ── Serialization → LLM context ────────────────────────────

    def to_llm_context(self) -> list[str]:
        """Serialize NPC knowledge into natural Skyrim vocabulary for LLM prompts."""
        lines: list[str] = []

        current_region = get_natural_position_name(
            self.npc_pos[0], self.npc_pos[1], self.world_size
        )
        lines.append(f"I am currently in {current_region}.")
        if self.learned_observed_cells:
            lines.append(
                f"I have personally explored {self.direct_coverage:.0%} of this region during my travels."
            )
            lines.append(
                f"Counting information learned from other NPCs, I know about {self.coverage:.0%} of this region."
            )
        else:
            lines.append(f"I've explored {self.coverage:.0%} of this region during my travels.")

        direct_regions = self.direct_explored_regions
        direct_explored = [r for r, v in direct_regions.items() if v]
        direct_unexplored = [r for r, v in direct_regions.items() if not v]
        if direct_explored:
            lines.append(f"I have traveled through the {', '.join(direct_explored)} area(s).")
        if direct_unexplored:
            lines.append(f"I have not yet ventured into the {', '.join(direct_unexplored)} region(s).")

        learned_regions = [
            r for r, v in self.learned_regions.items()
            if v and not direct_regions.get(r, False)
        ]
        if learned_regions:
            lines.append(f"Other NPCs have told me about the {', '.join(learned_regions)} area(s).")

        for npc_id in sorted(self.known_npc_goals):
            goal_label, kind, source_npc_ids = self.known_npc_goals[npc_id]
            natural_goal_name = get_natural_object_name(goal_label)
            npc_name = _npc_display_name(npc_id)
            if kind == "interaction":
                source_phrase = _goal_source_phrase(npc_id, source_npc_ids)
                source_phrase = source_phrase[0].upper() + source_phrase[1:]
                if len(source_npc_ids) == 1 and source_npc_ids[0] == npc_id:
                    lines.append(
                        f"{source_phrase} told me they are searching for the {natural_goal_name}."
                    )
                else:
                    lines.append(
                        f"{source_phrase} told me that {npc_name} is searching for the {natural_goal_name}."
                    )
            elif kind == "perfect":
                lines.append(f"I know that {npc_name} is searching for the {natural_goal_name}.")
            else:
                lines.append(f"I know that {npc_name} is searching for the {natural_goal_name}.")

        for label, positions in self.shape_locations.items():
            natural_name = get_natural_object_name(label)
            direct_positions: list[tuple[int, int]] = []
            interaction_positions_by_source: dict[tuple[str, ...], list[tuple[int, int]]] = {}
            perfect_positions: list[tuple[int, int]] = []

            for pos in positions:
                kind, npc_ids = self.observed_shape_sources.get(pos, ("direct", ()))
                if kind == "interaction":
                    interaction_positions_by_source.setdefault(npc_ids, []).append(pos)
                elif kind == "perfect":
                    perfect_positions.append(pos)
                else:
                    direct_positions.append(pos)

            if direct_positions:
                natural_locations = [
                    get_natural_position_name(x, y, self.world_size) for x, y in direct_positions
                ]
                if len(direct_positions) == 1:
                    lines.append(f"I found a {natural_name} in {natural_locations[0]}.")
                else:
                    location_list = ", ".join(natural_locations)
                    lines.append(f"I've seen {len(direct_positions)} {natural_name}s in: {location_list}.")

            for npc_ids, source_positions in sorted(interaction_positions_by_source.items()):
                natural_locations = [
                    get_natural_position_name(x, y, self.world_size) for x, y in source_positions
                ]
                source_phrase = _source_phrase(npc_ids)
                if len(source_positions) == 1:
                    lines.append(
                        f"{source_phrase} told me there is a {natural_name} in {natural_locations[0]}."
                    )
                else:
                    location_list = ", ".join(natural_locations)
                    lines.append(
                        f"{source_phrase} told me about {len(source_positions)} {natural_name}s in: {location_list}."
                    )

            if perfect_positions:
                natural_locations = [
                    get_natural_position_name(x, y, self.world_size) for x, y in perfect_positions
                ]
                if len(perfect_positions) == 1:
                    lines.append(f"I know there is a {natural_name} in {natural_locations[0]}.")
                else:
                    location_list = ", ".join(natural_locations)
                    lines.append(f"I know of {len(perfect_positions)} {natural_name}s in: {location_list}.")

        return lines
