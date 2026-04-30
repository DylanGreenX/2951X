from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import random

import config
from entities import NPC, Shape
from npc_brain import NPCBrain, NPCBrainGoalDriven, NPCBrainWandering
from world import GameWorld


@dataclass
class NPCActor:
    npc_id: str
    npc: NPC
    brain: NPCBrain

    @property
    def display_name(self) -> str:
        return f"NPC {int(self.npc_id.split('_')[-1]) + 1}"


@dataclass
class PartyTickResult:
    event_msgs: list[dict[str, str]] = field(default_factory=list)
    knowledge_exchanges: list[dict[str, object]] = field(default_factory=list)


def _all_goal_labels() -> list[str]:
    return [f"{color}_{shape}" for color in config.COLORS for shape in config.SHAPES]


def _default_spawn_positions(
    count: int,
    world_size: int,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    margin = 1 if world_size > 2 else 0
    far = max(0, world_size - 1 - margin)
    mid = world_size // 2
    candidates = [
        getattr(config, "NPC_START", (margin, margin)),
        (far, margin),
        (margin, far),
        (far, far),
        (mid, margin),
        (margin, mid),
        (far, mid),
        (mid, far),
        (mid, mid),
    ]

    starts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pos in candidates:
        if pos in reserved or pos in seen:
            continue
        x, y = pos
        if not (0 <= x < world_size and 0 <= y < world_size):
            continue
        starts.append(pos)
        seen.add(pos)
        if len(starts) == count:
            return starts

    for y in range(world_size):
        for x in range(world_size):
            pos = (x, y)
            if pos in reserved or pos in seen:
                continue
            starts.append(pos)
            seen.add(pos)
            if len(starts) == count:
                return starts
    raise ValueError(f"Unable to place {count} NPCs on a {world_size}x{world_size} grid")


class NPCParty:
    def __init__(
        self,
        actors: list[NPCActor],
        *,
        world: GameWorld,
        multiple_knowledge_mode: str,
    ) -> None:
        if not actors:
            raise ValueError("NPCParty requires at least one actor")
        if multiple_knowledge_mode not in {"shared", "independent"}:
            raise ValueError(
                "NPC_MULTIPLE_KNOWLEDGE_MODE must be 'shared' or 'independent'"
            )
        self.actors = actors
        self.world = world
        self.multiple_knowledge_mode = multiple_knowledge_mode

    @classmethod
    def from_config(cls, world: GameWorld, player_target_label: str) -> "NPCParty":
        count = int(getattr(config, "NPC_COUNT", 1))
        if count <= 0:
            raise ValueError(f"NPC_COUNT must be positive, got {count}")

        starts = cls._resolve_starts(world, count)
        goal_labels = cls._resolve_goal_labels(count, player_target_label)

        actors: list[NPCActor] = []
        for index, (start, goal_label) in enumerate(zip(starts, goal_labels, strict=True)):
            npc = NPC(start[0], start[1], sight_range=config.NPC_SIGHT_RANGE)
            if goal_label is None:
                brain = NPCBrainWandering(npc, world)
            else:
                brain = NPCBrainGoalDriven(npc, world, goal_label=goal_label)
            brain.on_state_updated()
            actors.append(NPCActor(npc_id=f"npc_{index}", npc=npc, brain=brain))

        party = cls(
            actors,
            world=world,
            multiple_knowledge_mode=getattr(
                config, "NPC_MULTIPLE_KNOWLEDGE_MODE", "shared"
            ),
        )
        if getattr(config, "NPC_KNOWLEDGE_MODE", "embodied") == "perfect":
            party.grant_perfect_knowledge()
        elif party.shared_knowledge:
            party.sync_shared_knowledge()
        return party

    @staticmethod
    def _resolve_starts(world: GameWorld, count: int) -> list[tuple[int, int]]:
        starts_cfg = getattr(config, "NPC_STARTS", None)
        reserved = {tuple(config.PLAYER_START)}
        if starts_cfg is not None:
            starts = [tuple(pos) for pos in starts_cfg]
            if len(starts) < count:
                raise ValueError(
                    f"NPC_STARTS only provides {len(starts)} positions for NPC_COUNT={count}"
                )
            starts = starts[:count]
        elif count == 1:
            starts = [tuple(config.NPC_START)]
        else:
            starts = _default_spawn_positions(count, world.size, reserved)

        seen: set[tuple[int, int]] = set()
        for start in starts:
            if start in seen:
                raise ValueError(f"Duplicate NPC start position {start}")
            seen.add(start)
            if not world.in_bounds(start[0], start[1]):
                raise ValueError(
                    f"NPC start {start} is out of bounds for world size {world.size}"
                )
            if start == tuple(config.PLAYER_START):
                raise ValueError("NPC start positions may not overlap PLAYER_START")
        return starts

    @classmethod
    def _resolve_goal_labels(cls, count: int, player_target_label: str) -> list[str | None]:
        if not getattr(config, "NPC_GOAL", True):
            return [None] * count

        competing_count_raw = getattr(config, "NPC_COMPETING_COUNT", None)
        if competing_count_raw is None:
            competing_count = count if getattr(config, "NPC_COMPETING", False) else 0
        else:
            competing_count = int(competing_count_raw)
        if not 0 <= competing_count <= count:
            raise ValueError(
                f"NPC_COMPETING_COUNT must be between 0 and NPC_COUNT ({count}), got {competing_count}"
            )

        goals: list[str | None] = [player_target_label] * competing_count
        non_competing_count = count - competing_count
        if non_competing_count == 0:
            return goals

        non_competing_goals = cls._resolve_non_competing_goals(
            non_competing_count, player_target_label
        )
        goals.extend(non_competing_goals)
        return goals

    @staticmethod
    def _resolve_non_competing_goals(
        count: int,
        player_target_label: str,
    ) -> list[str]:
        goal_mode = getattr(config, "NPC_NONCOMPETING_GOAL_MODE", "shared")
        if goal_mode not in {"shared", "unique"}:
            raise ValueError(
                "NPC_NONCOMPETING_GOAL_MODE must be 'shared' or 'unique'"
            )

        configured_label = f"{config.NPC_GOAL_COLOR}_{config.NPC_GOAL_SHAPE}"
        available = [label for label in _all_goal_labels() if label != player_target_label]
        if configured_label not in available and getattr(config, "NPC_GOAL_DETERMINISTIC", True):
            raise ValueError(
                "Configured non-competing goal matches the player's target; "
                "set NPC_COMPETING_COUNT or choose a different NPC_GOAL_* pair"
            )

        if goal_mode == "shared":
            if getattr(config, "NPC_GOAL_DETERMINISTIC", True):
                return [configured_label] * count
            return [random.choice(available)] * count

        if count > len(available):
            raise ValueError(
                f"Not enough distinct non-player goals for {count} non-competing NPCs"
            )

        if getattr(config, "NPC_GOAL_DETERMINISTIC", True):
            ordered = [configured_label] + sorted(
                [label for label in available if label != configured_label]
            )
            return ordered[:count]

        shuffled = available[:]
        random.shuffle(shuffled)
        return shuffled[:count]

    @property
    def shared_knowledge(self) -> bool:
        return len(self.actors) > 1 and self.multiple_knowledge_mode == "shared"

    @property
    def independent_knowledge(self) -> bool:
        return len(self.actors) > 1 and self.multiple_knowledge_mode == "independent"

    @property
    def brain_map(self) -> dict[str, NPCBrain]:
        return {actor.npc_id: actor.brain for actor in self.actors}

    def actor_by_id(self, npc_id: str | None) -> NPCActor:
        if npc_id is None:
            return self.actors[0]
        for actor in self.actors:
            if actor.npc_id == npc_id:
                return actor
        return self.actors[0]

    def next_actor_id(self, current_npc_id: str | None) -> str:
        if current_npc_id is None:
            return self.actors[0].npc_id
        ids = [actor.npc_id for actor in self.actors]
        if current_npc_id not in ids:
            return ids[0]
        index = ids.index(current_npc_id)
        return ids[(index + 1) % len(ids)]

    def actor_at(self, x: int, y: int) -> NPCActor | None:
        for actor in self.actors:
            if actor.npc.x == x and actor.npc.y == y:
                return actor
        return None

    def tick(self) -> PartyTickResult:
        result = PartyTickResult()

        if self.shared_knowledge:
            self.sync_shared_knowledge()
        else:
            result.knowledge_exchanges.extend(self._run_local_exchanges())

        for actor in self.actors:
            event_msg = actor.brain.tick()
            if event_msg:
                result.event_msgs.append(
                    {
                        "npc_id": actor.npc_id,
                        "display_name": actor.display_name,
                        "message": event_msg,
                    }
                )

        if self.shared_knowledge:
            self.sync_shared_knowledge()
        else:
            result.knowledge_exchanges.extend(self._run_local_exchanges())

        return result

    def combined_observed_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for actor in self.actors:
            cells |= actor.brain.state.observed_cells
        return cells

    def combined_coverage(self) -> float:
        return len(self.combined_observed_cells()) / (self.world.size ** 2)

    def grant_perfect_knowledge(self) -> None:
        observed_cells = {
            (x, y)
            for x in range(self.world.size)
            for y in range(self.world.size)
        }
        observed_shapes = list(self.world.shapes)
        shape_locations: dict[str, list[tuple[int, int]]] = {}
        shape_first_tick: dict[tuple[int, int], int] = {}
        observed_cell_sources = {
            (x, y): ("perfect", ())
            for x in range(self.world.size)
            for y in range(self.world.size)
        }
        observed_shape_sources: dict[tuple[int, int], tuple[str, tuple[str, ...]]] = {}
        known_npc_goals = {
            actor.npc_id: (actor.npc.goal_label, "perfect", ())
            for actor in self.actors
            if getattr(actor.npc, "goal_label", None)
        }
        for shape in self.world.shapes:
            shape_locations.setdefault(shape.label, []).append((shape.x, shape.y))
            shape_first_tick[(shape.x, shape.y)] = 0
            observed_shape_sources[(shape.x, shape.y)] = ("perfect", ())

        for actor in self.actors:
            actor.brain.state.replace_memory(
                observed_cells=observed_cells,
                observed_shapes=observed_shapes,
                shape_locations=shape_locations,
                shape_first_tick=shape_first_tick,
                observed_cell_sources=observed_cell_sources,
                observed_shape_sources=observed_shape_sources,
                known_npc_goals=known_npc_goals,
            )
            actor.brain.on_state_updated()

    def sync_shared_knowledge(self) -> None:
        observed_cells: set[tuple[int, int]] = set()
        observed_shapes_by_pos: dict[tuple[int, int], Shape] = {}
        shape_locations_set: dict[str, set[tuple[int, int]]] = {}
        shape_first_tick: dict[tuple[int, int], int] = {}

        for actor in self.actors:
            state = actor.brain.state
            observed_cells |= state.observed_cells
            for shape in state.observed_shapes:
                observed_shapes_by_pos[(shape.x, shape.y)] = shape
            for label, positions in state.shape_locations.items():
                shape_locations_set.setdefault(label, set()).update(positions)
            for loc, tick in state.shape_first_tick.items():
                shape_first_tick[loc] = min(shape_first_tick.get(loc, tick), tick)

        self._apply_memory_snapshot(
            self.actors,
            observed_cells=observed_cells,
            observed_shapes_by_pos=observed_shapes_by_pos,
            shape_locations_set=shape_locations_set,
            shape_first_tick=shape_first_tick,
        )

    def _run_local_exchanges(self) -> list[dict[str, object]]:
        if not self.independent_knowledge:
            return []
        if not getattr(config, "NPC_NPC_INTERACTION_ENABLED", True):
            return []

        events: list[dict[str, object]] = []
        for actor_a, actor_b in combinations(self.actors, 2):
            if not self._can_exchange(actor_a, actor_b):
                continue
            event = self._exchange_pair(actor_a, actor_b)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _can_exchange(actor_a: NPCActor, actor_b: NPCActor) -> bool:
        dx = abs(actor_a.npc.x - actor_b.npc.x)
        dy = abs(actor_a.npc.y - actor_b.npc.y)
        chebyshev = max(dx, dy)
        return chebyshev <= actor_a.npc.sight_range or chebyshev <= actor_b.npc.sight_range

    def _exchange_pair(self, actor_a: NPCActor, actor_b: NPCActor) -> dict[str, object] | None:
        before_a = self._memory_signature(actor_a)
        before_b = self._memory_signature(actor_b)

        observed_cells = set(actor_a.brain.state.observed_cells) | set(actor_b.brain.state.observed_cells)
        observed_shapes_by_pos: dict[tuple[int, int], Shape] = {}
        shape_locations_set: dict[str, set[tuple[int, int]]] = {}
        shape_first_tick: dict[tuple[int, int], int] = {}

        for actor in (actor_a, actor_b):
            state = actor.brain.state
            for shape in state.observed_shapes:
                observed_shapes_by_pos[(shape.x, shape.y)] = shape
            for label, positions in state.shape_locations.items():
                shape_locations_set.setdefault(label, set()).update(positions)
            for loc, tick in state.shape_first_tick.items():
                shape_first_tick[loc] = min(shape_first_tick.get(loc, tick), tick)

        self._apply_memory_snapshot(
            [actor_a, actor_b],
            observed_cells=observed_cells,
            observed_shapes_by_pos=observed_shapes_by_pos,
            shape_locations_set=shape_locations_set,
            shape_first_tick=shape_first_tick,
        )

        after_a = self._memory_signature(actor_a)
        after_b = self._memory_signature(actor_b)
        if before_a == after_a and before_b == after_b:
            return None

        return {
            "participants": [actor_a.npc_id, actor_b.npc_id],
            "participant_names": [actor_a.display_name, actor_b.display_name],
            "positions": [
                [actor_a.npc.x, actor_a.npc.y],
                [actor_b.npc.x, actor_b.npc.y],
            ],
            "shared_cells_count": len(observed_cells),
        }

    @staticmethod
    def _apply_memory_snapshot(
        actors: list[NPCActor],
        *,
        observed_cells: set[tuple[int, int]],
        observed_shapes_by_pos: dict[tuple[int, int], Shape],
        shape_locations_set: dict[str, set[tuple[int, int]]],
        shape_first_tick: dict[tuple[int, int], int],
    ) -> None:
        observed_shapes = [
            observed_shapes_by_pos[pos] for pos in sorted(observed_shapes_by_pos)
        ]
        shape_locations = {
            label: sorted(positions)
            for label, positions in shape_locations_set.items()
        }

        for actor in actors:
            observed_cell_sources = {
                cell: NPCParty._merged_source_for_cell(actors, actor, cell)
                for cell in observed_cells
            }
            observed_shape_sources = {
                pos: NPCParty._merged_source_for_shape(actors, actor, pos)
                for pos in observed_shapes_by_pos
            }
            known_npc_goals = NPCParty._merged_goal_knowledge(actors, actor)
            actor.brain.state.replace_memory(
                observed_cells=observed_cells,
                observed_shapes=observed_shapes,
                shape_locations=shape_locations,
                shape_first_tick=shape_first_tick,
                observed_cell_sources=observed_cell_sources,
                observed_shape_sources=observed_shape_sources,
                known_npc_goals=known_npc_goals,
            )
            actor.brain.on_state_updated()

    @staticmethod
    def _memory_signature(actor: NPCActor) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[str, int, int]]]:
        shape_positions = frozenset(
            (label, x, y)
            for label, positions in actor.brain.state.shape_locations.items()
            for (x, y) in positions
        )
        return frozenset(actor.brain.state.observed_cells), shape_positions

    @staticmethod
    def _merged_source_for_cell(
        actors: list[NPCActor],
        target_actor: NPCActor,
        cell: tuple[int, int],
    ) -> tuple[str, tuple[str, ...]]:
        current = target_actor.brain.state.observed_cell_sources.get(cell)
        if current is not None and current[0] in {"direct", "perfect"}:
            return current

        informants: set[str] = set()
        saw_perfect = False
        for actor in actors:
            source = actor.brain.state.observed_cell_sources.get(cell)
            if source is None:
                if cell in actor.brain.state.observed_cells:
                    informants.add(actor.npc_id)
                continue
            kind, npc_ids = source
            if kind == "perfect":
                saw_perfect = True
            elif kind == "direct":
                informants.add(actor.npc_id)
            elif kind == "interaction":
                informants.update(npc_ids)

        informants.discard(target_actor.npc_id)
        if informants:
            return ("interaction", tuple(sorted(informants)))
        if current is not None:
            return current
        if saw_perfect:
            return ("perfect", ())
        return ("direct", ())

    @staticmethod
    def _merged_source_for_shape(
        actors: list[NPCActor],
        target_actor: NPCActor,
        pos: tuple[int, int],
    ) -> tuple[str, tuple[str, ...]]:
        current = target_actor.brain.state.observed_shape_sources.get(pos)
        if current is not None and current[0] in {"direct", "perfect"}:
            return current

        informants: set[str] = set()
        saw_perfect = False
        for actor in actors:
            source = actor.brain.state.observed_shape_sources.get(pos)
            if source is None:
                continue
            kind, npc_ids = source
            if kind == "perfect":
                saw_perfect = True
            elif kind == "direct":
                informants.add(actor.npc_id)
            elif kind == "interaction":
                informants.update(npc_ids)

        informants.discard(target_actor.npc_id)
        if informants:
            return ("interaction", tuple(sorted(informants)))
        if current is not None:
            return current
        if saw_perfect:
            return ("perfect", ())
        return ("direct", ())

    @staticmethod
    def _merged_goal_knowledge(
        actors: list[NPCActor],
        target_actor: NPCActor,
    ) -> dict[str, tuple[str, str, tuple[str, ...]]]:
        knowledge = dict(target_actor.brain.state.known_npc_goals)

        for actor in actors:
            if actor.npc_id == target_actor.npc_id:
                continue

            goal_label = getattr(actor.npc, "goal_label", None)
            if goal_label:
                knowledge[actor.npc_id] = (goal_label, "interaction", (actor.npc_id,))

            for known_npc_id, (known_goal_label, kind, source_npc_ids) in actor.brain.state.known_npc_goals.items():
                if known_npc_id == target_actor.npc_id or not known_goal_label:
                    continue
                if kind == "perfect":
                    knowledge.setdefault(known_npc_id, (known_goal_label, "perfect", ()))
                    continue

                merged_sources = set(source_npc_ids)
                merged_sources.add(actor.npc_id)
                existing = knowledge.get(known_npc_id)
                if existing is not None and existing[1] == "perfect":
                    continue
                knowledge[known_npc_id] = (
                    known_goal_label,
                    "interaction",
                    tuple(sorted(merged_sources)),
                )

        knowledge.pop(target_actor.npc_id, None)
        return knowledge
