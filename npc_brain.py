"""
NPC exploration brain.

The NPC has its OWN goal: collect blue circles.
Its knowledge about other shapes (including the player's target,
the red triangle) is a side effect of pursuing that goal.

Current policy: greedy move toward nearest known blue circle,
with random exploration when no target is known.
This is a stand-in for a trained RL agent — produces the same
kind of output (goal-directed movement + incidental observations).
"""
import random
from entities import NPC, Shape
from world import GameWorld
from rlang_engine import RLangState


class NPCBrain:
    def __init__(self, npc: NPC, world: GameWorld):
        self.npc = npc
        self.world = world
        self.state = RLangState(world_size=world.size)

        # Exploration memory for smarter wandering
        self._recent_positions: list[tuple[int, int]] = []
        self._max_recent = 10

    def tick(self) -> str | None:
        """
        One NPC step. Returns an event string if something notable happened.
        """
        # 1. OBSERVE — sense within sight range
        visible = self.world.get_visible_cells(
            self.npc.x, self.npc.y, self.npc.sight_range
        )
        self.state.observe(visible)

        # 2. DECIDE — pick direction based on goal
        direction = self._choose_direction()

        # 3. ACT — move
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
        nx, ny = self.npc.x + dx, self.npc.y + dy

        if self.world.in_bounds(nx, ny):
            self.npc.x = nx
            self.npc.y = ny

        self.npc.steps_taken += 1
        self.state.npc_pos = (self.npc.x, self.npc.y)  # update after move
        self._recent_positions.append((self.npc.x, self.npc.y))
        if len(self._recent_positions) > self._max_recent:
            self._recent_positions.pop(0)

        # 4. OBSERVE AGAIN after moving (see new cell)
        visible = self.world.get_visible_cells(
            self.npc.x, self.npc.y, self.npc.sight_range
        )
        self.state.observe(visible)

        # 5. CHECK — did we reach a blue circle?
        shape = self.world.shape_at(self.npc.x, self.npc.y)
        if shape and shape.label == "blue_circle" and not shape.collected:
            shape.collected = True
            self.state.record_collection(shape)
            self.npc.blue_circles_collected += 1
            return f"Collected blue circle at ({shape.x}, {shape.y})!"

        return None

    def _choose_direction(self) -> str:
        """
        Greedy policy: move toward nearest known uncollected blue circle.
        If none known, explore (biased random walk away from recent cells).

        This is a placeholder for a trained RL policy. The interface
        (observe → decide → act) stays the same when swapping to RL.
        """
        # Goal-directed: move toward nearest known blue circle
        if self.state.known_blue_circle_positions:
            target = self._nearest_blue_circle()
            if target:
                return self._direction_toward(target)

        # Exploration: biased random walk favoring unvisited areas
        return self._explore_direction()

    def _nearest_blue_circle(self) -> tuple[int, int] | None:
        best = None
        best_dist = float("inf")
        for pos in self.state.known_blue_circle_positions:
            # Verify it's still there (not collected)
            shape = self.world.shape_at(pos[0], pos[1])
            if shape and shape.label == "blue_circle" and not shape.collected:
                d = abs(pos[0] - self.npc.x) + abs(pos[1] - self.npc.y)
                if d < best_dist:
                    best_dist = d
                    best = pos
        # Clean up stale positions
        if best is None:
            self.state.known_blue_circle_positions.clear()
        return best

    def _direction_toward(self, target: tuple[int, int]) -> str:
        dx = target[0] - self.npc.x
        dy = target[1] - self.npc.y

        # Prefer larger delta axis, with small random tie-breaking
        options = []
        if dx > 0: options.append("right")
        elif dx < 0: options.append("left")
        if dy > 0: options.append("down")
        elif dy < 0: options.append("up")

        return random.choice(options) if options else random.choice(["up", "down", "left", "right"])

    def _explore_direction(self) -> str:
        """Biased walk: prefer directions leading to less-visited areas."""
        directions = {
            "up": (0, -1), "down": (0, 1),
            "left": (-1, 0), "right": (1, 0)
        }

        scored = []
        for name, (dx, dy) in directions.items():
            nx, ny = self.npc.x + dx, self.npc.y + dy
            if not self.world.in_bounds(nx, ny):
                continue

            # Score: prefer unvisited cells, penalize recently visited
            score = 0
            if (nx, ny) not in self.state.observed_cells:
                score += 10
            if (nx, ny) in self._recent_positions:
                score -= 5
            score += random.random() * 3  # noise for variety

            scored.append((score, name))

        if not scored:
            return random.choice(["up", "down", "left", "right"])

        scored.sort(reverse=True)
        return scored[0][1]
