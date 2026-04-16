"""
Concrete GameAPIProvider for the pygame needle-in-haystack demo.

Wraps GameWorld, Player, and the NPC brain dict to satisfy every abstract
method defined in GameAPIProvider. All return values are plain TypedDicts
(JSON-serializable) so they can be passed directly into LLM tool-result
messages without any extra serialization step.

Wiring into the game
────────────────────
    from pygame_game_api import PygameGameAPI

    # Single-NPC convenience constructor
    api = PygameGameAPI.from_game(world, player, brain)

    # Multi-NPC explicit constructor
    api = PygameGameAPI(world, player, {"npc_0": brain_a, "npc_1": brain_b})

    # Pass to InteractionManager (once wired up) or your own dispatcher
    result = dispatch_tool_call(api, tool_name, args)

The brains dict is held by reference — mutations to the brain objects
(new observations, position updates) are reflected immediately in subsequent
API calls without needing to recreate the provider.
"""
from __future__ import annotations

from typing import Optional

from entities import Player
from game_api_interface import (
    AllObjectsDict,
    ExplorationStatusDict,
    GameAPIProvider,
    NPCMemoryDict,
    NPCStateDict,
    NearbyObjectsDict,
    ObjectAtDict,
    ObjectDict,
    PlayerStateDict,
    PositionDict,
    SetNPCTargetDict,
    WorldInfoDict,
)
from npc_brain import NPCBrain
from rlang_engine import get_natural_object_name
from world import GameWorld


class PygameGameAPI(GameAPIProvider):
    """
    Concrete GameAPIProvider for the pygame grid demo.

    Implements all eight GameAPIProvider methods by reading from the live
    GameWorld, Player entity, and NPCBrain instances. No game state is
    cached here — every call reads the current value directly.
    """

    DEFAULT_NPC_ID = "npc_0"

    def __init__(
        self,
        world: GameWorld,
        player: Player,
        brains: dict[str, NPCBrain],
    ) -> None:
        """
        Args:
            world:  The active GameWorld instance.
            player: The Player entity.
            brains: Mapping of npc_id → NPCBrain.
                    In the single-NPC demo use PygameGameAPI.from_game() instead.
        """
        self.world = world
        self.player = player
        self.brains = brains

    # ── Convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_game(
        cls,
        world: GameWorld,
        player: Player,
        brain: NPCBrain,
    ) -> "PygameGameAPI":
        """Build a provider for the single-NPC demo."""
        return cls(world, player, {cls.DEFAULT_NPC_ID: brain})

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_brain(self, npc_id: str) -> NPCBrain:
        if npc_id not in self.brains:
            raise KeyError(
                f"Unknown npc_id {npc_id!r}. "
                f"Registered IDs: {list(self.brains)}"
            )
        return self.brains[npc_id]

    @staticmethod
    def _to_object_dict(shape) -> ObjectDict:
        """Convert a Shape entity to a JSON-safe ObjectDict."""
        return {
            "label": shape.label,
            "natural_name": get_natural_object_name(shape.label),
            "position": {"x": shape.x, "y": shape.y},
            "collected": shape.collected,
        }

    # ── GameAPIProvider implementation ────────────────────────────────────────

    def get_world_info(self) -> WorldInfoDict:
        """
        Return grid dimensions and the player's quest target.

        The target information comes from the GameWorld, not from the NPC —
        this represents what the quest system knows, not what the NPC knows.
        """
        target_label = f"{self.world.target_color}_{self.world.target_shape}"
        return {
            "size": self.world.size,
            "target_label": target_label,
            "target_natural_name": get_natural_object_name(target_label),
        }

    def get_npc_state(self, npc_id: str) -> NPCStateDict:
        """
        Return position, goal, step count, and sight range for the given NPC.

        goal_label / goal_natural_name are None for wandering (baseline) NPCs.
        """
        brain = self._get_brain(npc_id)
        npc = brain.npc
        goal_label: Optional[str] = getattr(npc, "goal_label", None)
        return {
            "npc_id": npc_id,
            "position": {"x": npc.x, "y": npc.y},
            "sight_range": npc.sight_range,
            "steps_taken": npc.steps_taken,
            "goal_label": goal_label,
            "goal_natural_name": (
                get_natural_object_name(goal_label) if goal_label else None
            ),
        }

    def get_player_state(self) -> PlayerStateDict:
        return {
            "position": {"x": self.player.x, "y": self.player.y},
            "sight_range": self.player.sight_range,
        }

    def get_nearby_objects(self, x: int, y: int, radius: int = 3) -> NearbyObjectsDict:
        """
        Return uncollected objects within `radius` cells of (x, y).

        Uses GameWorld.get_visible_cells, the same visibility function the NPC
        brain uses during its observe step.
        """
        cells = self.world.get_visible_cells(x, y, radius)
        objects: list[ObjectDict] = [
            self._to_object_dict(shape)
            for _, _, shape in cells
            if shape is not None and not shape.collected
        ]
        return {
            "queried_position": {"x": x, "y": y},
            "radius": radius,
            "objects": objects,
        }

    def get_npc_memory(
        self,
        npc_id: str,
        filter_label: Optional[str] = None,
    ) -> NPCMemoryDict:
        """
        Return the NPC's embodied observation log.

        If filter_label is given (e.g. "red_triangle") only entries for that
        label are included. If the NPC has never seen that object the returned
        shape_locations dict will be empty — do not fall back to full world state.

        context_lines is the same list that to_llm_context() produces and can
        be injected directly into an LLM system or user message.
        """
        brain = self._get_brain(npc_id)
        state = brain.state

        # Filter or use the full observed map
        if filter_label is not None:
            raw: dict[str, list[tuple[int, int]]] = (
                {filter_label: state.shape_locations[filter_label]}
                if filter_label in state.shape_locations
                else {}
            )
        else:
            raw = dict(state.shape_locations)

        shape_locations: dict[str, list[PositionDict]] = {
            label: [{"x": x, "y": y} for x, y in positions]
            for label, positions in raw.items()
        }

        return {
            "npc_id": npc_id,
            "coverage": state.coverage,
            "shape_locations": shape_locations,
            "context_lines": state.to_llm_context(),
        }

    def get_exploration_status(self, npc_id: str) -> ExplorationStatusDict:
        """
        Return the NPC's spatial exploration breakdown.

        explored_regions maps quadrant names ("NW", "NE", "SW", "SE") to
        boolean — True if the NPC has visited at least one cell in that region.
        """
        brain = self._get_brain(npc_id)
        state = brain.state
        return {
            "npc_id": npc_id,
            "coverage": state.coverage,
            "explored_regions": state.explored_regions,
            "total_cells_observed": len(state.observed_cells),
        }

    def get_object_at(self, x: int, y: int) -> ObjectAtDict:
        """
        Return the object at (x, y), or None if the cell is empty.

        Uses GameWorld.shape_at, which already filters out collected objects.
        """
        shape = self.world.shape_at(x, y)
        return {
            "position": {"x": x, "y": y},
            "object": self._to_object_dict(shape) if shape is not None else None,
        }

    def get_all_objects(self) -> AllObjectsDict:
        """
        Return every uncollected object — full world state, no NPC filter.

        Only appropriate for the omniscient-NPC baseline experiment condition.
        """
        return {
            "objects": [
                self._to_object_dict(s)
                for s in self.world.shapes
                if not s.collected
            ]
        }

    def set_npc_target(self, npc_id: str, x: int, y: int) -> SetNPCTargetDict:
        """
        Command the NPC to navigate to (x, y) one step at a time.

        Validates that the npc_id exists and the coordinates are within the
        world bounds before calling set_target_pos on the brain. Returns
        success=False with a descriptive message if either check fails.
        """
        if npc_id not in self.brains:
            return {
                "npc_id": npc_id,
                "target_position": {"x": x, "y": y},
                "success": False,
                "message": (
                    f"Unknown npc_id {npc_id!r}. "
                    f"Registered IDs: {list(self.brains)}"
                ),
            }

        if not self.world.in_bounds(x, y):
            return {
                "npc_id": npc_id,
                "target_position": {"x": x, "y": y},
                "success": False,
                "message": (
                    f"Target ({x}, {y}) is out of bounds. "
                    f"World size is {self.world.size}x{self.world.size} "
                    f"(valid range: 0 to {self.world.size - 1})."
                ),
            }

        self.brains[npc_id].set_target_pos((x, y))
        return {
            "npc_id": npc_id,
            "target_position": {"x": x, "y": y},
            "success": True,
            "message": f"NPC {npc_id!r} is now navigating to ({x}, {y}).",
        }

