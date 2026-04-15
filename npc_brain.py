"""
NPC brains for wandering and goal-driven exploration.

NPCBrainWandering  — baseline: random biased walk, no goal. Accumulates
                     incidental observations about the world via RLangState.

NPCBrainGoalDriven — extension: pursues a specific target label (e.g.
                     "blue_circle"). Wanders until a target enters its sight
                     range, then navigates toward it greedily. Goal tracking
                     (what it has collected, where targets are) lives here on
                     the brain, not in RLangState, which stays goal-agnostic.

The observe → decide → act loop interface is stable across both subclasses,
making it easy to swap in an RL-trained policy later.
"""

import random
from entities import NPC
from world import GameWorld
from rlang_engine import RLangState, get_natural_object_name


class NPCBrain:
    def __init__(self, npc: NPC, world: GameWorld):
        self.npc = npc
        self.world = world
        self.state = RLangState(world_size=world.size)
        self._recent_positions: list[tuple[int, int]] = []
        self._max_recent = 10

    def tick(self) -> str | None:
        raise NotImplementedError

    def _choose_direction(self) -> str:
        raise NotImplementedError

    def _move(self, direction: str) -> None:
        dx, dy = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }[direction]

        nx, ny = self.npc.x + dx, self.npc.y + dy
        if self.world.in_bounds(nx, ny):
            self.npc.x = nx
            self.npc.y = ny

        self.npc.steps_taken += 1
        self.state.npc_pos = (self.npc.x, self.npc.y)

        self._recent_positions.append((self.npc.x, self.npc.y))
        if len(self._recent_positions) > self._max_recent:
            self._recent_positions.pop(0)

    def _observe(self) -> None:
        visible = self.world.get_visible_cells(
            self.npc.x, self.npc.y, self.npc.sight_range
        )
        self.state.observe(visible, current_step=self.npc.steps_taken)

    def _explore_direction(self) -> str:
        """Biased walk: prefer unvisited cells, avoid recently visited ones."""
        directions = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }

        scored = []
        for name, (dx, dy) in directions.items():
            nx, ny = self.npc.x + dx, self.npc.y + dy
            if not self.world.in_bounds(nx, ny):
                continue

            score = 0
            if (nx, ny) not in self.state.observed_cells:
                score += 10
            if (nx, ny) in self._recent_positions:
                score -= 5
            score += random.random() * 3
            scored.append((score, name))

        if not scored:
            return random.choice(list(directions.keys()))

        scored.sort(reverse=True)
        return scored[0][1]


class NPCBrainWandering(NPCBrain):
    """Baseline NPC: wanders aimlessly, accumulates observations."""

    def tick(self) -> str | None:
        self._observe()
        self._move(self._choose_direction())
        self._observe()
        return None

    def _choose_direction(self) -> str:
        return self._explore_direction()


class NPCBrainGoalDriven(NPCBrain):
    """
    Goal-driven NPC: pursues a specific shape label.

    The NPC wanders until it spots its target within sight range, then
    navigates toward the remembered location. Goal state (collection count,
    known target positions) is tracked here on the brain. RLangState only
    holds generic observations.

    Note: if NPC_COMPETING is True (set in config), the goal_label will
    match the player's target. The win condition in main.py gates whether
    the NPC collecting the target ends the game.
    """

    def __init__(self, npc: NPC, world: GameWorld, goal_label: str):
        super().__init__(npc, world)
        # goal_label is also stored on the entity for logging / replay access
        self.npc.goal_label = goal_label
        self.goal_label = goal_label
        self.natural_goal_name = get_natural_object_name(goal_label)
        self.goal_collected: int = 0
        # Known uncollected positions of the target, learned from observations
        self._known_target_positions: list[tuple[int, int]] = []

    def tick(self) -> str | None:
        self._observe()
        self._sync_known_targets()
        self._move(self._choose_direction())
        self._observe()
        self._sync_known_targets()

        shape = self.world.shape_at(self.npc.x, self.npc.y)
        if shape and shape.label == self.goal_label and not shape.collected:
            shape.collected = True
            self.goal_collected += 1
            pos = (shape.x, shape.y)
            if pos in self._known_target_positions:
                self._known_target_positions.remove(pos)
            return f"Collected {self.natural_goal_name} at ({shape.x}, {shape.y})!"

        return None

    def _choose_direction(self) -> str:
        target = self._nearest_known_target()
        if target:
            return self._direction_toward(target)
        return self._explore_direction()

    def _sync_known_targets(self) -> None:
        """
        Pull newly observed target positions from RLangState into our local
        tracking list (deduplicated, and cleaned of stale collected entries).
        """
        seen = self.state.shape_locations.get(self.goal_label, [])
        for pos in seen:
            shape = self.world.shape_at(pos[0], pos[1])
            if shape and not shape.collected and pos not in self._known_target_positions:
                self._known_target_positions.append(pos)
        # Remove positions that have since been collected
        self._known_target_positions = [
            pos for pos in self._known_target_positions
            if (s := self.world.shape_at(pos[0], pos[1])) and not s.collected
        ]

    def _nearest_known_target(self) -> tuple[int, int] | None:
        best = None
        best_dist = float("inf")
        for pos in self._known_target_positions:
            d = abs(pos[0] - self.npc.x) + abs(pos[1] - self.npc.y)
            if d < best_dist:
                best_dist = d
                best = pos
        return best

    def _direction_toward(self, target: tuple[int, int]) -> str:
        dx = target[0] - self.npc.x
        dy = target[1] - self.npc.y

        options = []
        if dx > 0:
            options.append("right")
        elif dx < 0:
            options.append("left")
        if dy > 0:
            options.append("down")
        elif dy < 0:
            options.append("up")

        return random.choice(options) if options else random.choice(["up", "down", "left", "right"])
