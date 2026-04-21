"""
Game API Provider: abstract interface, return-type contracts, and LLM tool schemas.

Any game that wants to connect NPC brains to the LLM tool-calling layer must
provide a concrete subclass of GameAPIProvider. The LLMToolDispatcher in
interaction.py calls these methods and passes results back to the LLM as
structured tool results.

Return types are TypedDicts: plain dicts at runtime (JSON-serializable and
ready to pass directly to LLM tool-result messages) while still being
statically type-checked by IDEs / mypy.

Usage pattern
─────────────
  1. Subclass GameAPIProvider and implement every abstract method.
  2. Pass your instance to InteractionManager (or your own dispatcher).
  3. Convert GAME_TOOL_SCHEMAS to provider-native tool declarations.
  4. Dispatch incoming function calls with dispatch_tool_call().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

# ── Shared sub-types ──────────────────────────────────────────────────────────

from typing import TypedDict


class PositionDict(TypedDict):
    x: int
    y: int


class ObjectDict(TypedDict):
    """LLM-facing object record. Internal labels like 'red_triangle' never
    appear here — the LLM reasons about items by their Skyrim natural name
    (e.g. 'crimson flag') and acts on them by position."""
    name: str           # Skyrim vocabulary name, e.g. "crimson flag"
    position: PositionDict
    region: str         # natural region phrase, e.g. "the far southeast corner"
    collected: bool


# ── Per-method return types ───────────────────────────────────────────────────

class WorldInfoDict(TypedDict):
    """Static world metadata."""
    size: int                    # grid is size × size
    target_name: str             # Skyrim natural name, e.g. "crimson flag"


class NPCStateDict(TypedDict):
    """Live state of a single NPC."""
    npc_id: str
    position: PositionDict
    region: str                    # natural region phrase for position
    sight_range: int
    steps_taken: int
    goal_name: Optional[str]       # Skyrim natural name, None for wandering NPC


class PlayerStateDict(TypedDict):
    """Live state of the player."""
    position: PositionDict
    sight_range: int


class NearbyObjectsDict(TypedDict):
    """Objects found within a radius search."""
    queried_position: PositionDict
    radius: int
    objects: list[ObjectDict]


class NPCMemoryDict(TypedDict):
    """Everything the NPC personally observed. Keyed by Skyrim natural name
    (e.g. 'crimson flag'), never by internal label — the LLM only ever sees
    one identifier for each object."""
    npc_id: str
    coverage: float                              # fraction of world explored, 0.0–1.0
    observations: dict[str, list[PositionDict]]  # natural name → list of positions seen
    context_lines: list[str]                     # natural-language sentences for LLM injection


class ExplorationStatusDict(TypedDict):
    """Spatial breakdown of the NPC's exploration progress."""
    npc_id: str
    coverage: float
    explored_regions: dict[str, bool]  # {"NW": True, "NE": False, "SW": True, "SE": False}
    total_cells_observed: int


class ObjectAtDict(TypedDict):
    """Result of querying a specific cell."""
    position: PositionDict
    object: Optional[ObjectDict]   # None → cell is empty


class AllObjectsDict(TypedDict):
    """Full world object list (perfect-knowledge mode only)."""
    objects: list[ObjectDict]


class SetNPCTargetDict(TypedDict):
    """Result of issuing a movement target to an NPC."""
    npc_id: str
    target_position: PositionDict
    success: bool
    message: str


# ── Abstract base class ───────────────────────────────────────────────────────

class GameAPIProvider(ABC):
    """
    Abstract interface every game must implement to connect NPC brains to
    the LLM tool-calling layer.

    One concrete subclass per game (e.g. PygameGameAPI, UnrealGameAPI).
    All methods return TypedDicts so results are JSON-serializable and can
    be passed directly as tool results in LLM API message lists.

    Enforcement: any subclass that omits an abstract method raises TypeError
    at instantiation time — not silently at call time.
    """

    @abstractmethod
    def get_world_info(self) -> WorldInfoDict:
        """
        Return static world metadata: dimensions and the player's quest target.
        Call once at conversation start so the NPC understands the overall context.
        """
        ...

    @abstractmethod
    def get_npc_state(self, npc_id: str) -> NPCStateDict:
        """
        Return current position, goal, steps taken, and sight range for the
        given NPC. Use when the NPC needs to reason about its own status or
        explain how much of the world it has been able to observe.
        """
        ...

    @abstractmethod
    def get_player_state(self) -> PlayerStateDict:
        """
        Return the player's current position and sight range. Use when
        reasoning about proximity to a target or directing the player.
        """
        ...

    @abstractmethod
    def get_nearby_objects(self, x: int, y: int, radius: int = 3) -> NearbyObjectsDict:
        """
        Return all uncollected game objects within `radius` grid cells of (x, y).
        Use when the player asks what items are near a particular location or
        when verifying whether a remembered object is still present.
        """
        ...

    @abstractmethod
    def get_npc_memory(
        self, npc_id: str, filter_name: Optional[str] = None
    ) -> NPCMemoryDict:
        """
        Return the NPC's personal observation log — only items the NPC has
        personally witnessed during its travels. Never returns the full game
        state; this is embodied knowledge only.

        Pass filter_name (e.g. "crimson flag") to narrow results to one object
        type by Skyrim natural name. Omit it to retrieve the full history.
        """
        ...

    @abstractmethod
    def get_exploration_status(self, npc_id: str) -> ExplorationStatusDict:
        """
        Return what fraction of the world the NPC has explored and which
        quadrant regions it has visited. Use when the player asks whether
        the NPC has been to a specific area or to express uncertainty about
        unexplored zones.
        """
        ...

    @abstractmethod
    def get_object_at(self, x: int, y: int) -> ObjectAtDict:
        """
        Return the object at a specific grid position, or None if the cell
        is empty (object was already collected or never placed there). Use
        for precise location lookups when the NPC remembers an exact spot.
        """
        ...

    @abstractmethod
    def get_all_objects(self) -> AllObjectsDict:
        """
        Return every uncollected object in the world regardless of NPC
        observations. This is *perfect-knowledge mode* — call only when
        running the omniscient NPC baseline condition. Never use this to
        generate embodied NPC responses.
        """
        ...

    @abstractmethod
    def set_npc_target(self, npc_id: str, x: int, y: int) -> SetNPCTargetDict:
        """
        Command the NPC to autonomously navigate to grid position (x, y).
        The NPC will step one cell at a time toward the target each game tick
        until it arrives, then resume its default behaviour. Use when you want
        the NPC to physically move to a location — for example, to lead the
        player to a known object, investigate a position, or reposition before
        speaking again.
        """
        ...


# ── LLM tool schemas ──────────────────────────────────────────────────────────
# GAME_TOOL_SCHEMAS uses an OpenAI-shaped function schema because it is compact
# and provider-neutral. llm.LLMClient converts these declarations to Gemini
# Tool objects before model calls. The descriptions tell the model when to call
# each tool and what it will receive back.

GAME_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_world_info",
            "description": (
                "Returns the world's grid dimensions and the player's quest target object. "
                "Call once at the start of a conversation to understand the overall context."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_npc_state",
            "description": (
                "Returns the NPC's current position, goal label, steps taken, and sight range. "
                "Use when the NPC needs to describe its own situation or movement history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_id": {
                        "type": "string",
                        "description": "The NPC identifier, e.g. 'npc_0'.",
                    }
                },
                "required": ["npc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_state",
            "description": (
                "Returns the player's current position and sight range. "
                "Use when reasoning about where to direct the player."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_objects",
            "description": (
                "Returns all uncollected game objects within a radius of a map position. "
                "Use when the player asks what items are near a specific location, "
                "or to verify that a remembered object has not yet been collected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Grid X coordinate (0 to world_size - 1).",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Grid Y coordinate (0 to world_size - 1).",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Search radius in grid cells. Typical values: 1–5. Defaults to 3.",
                        "default": 3,
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_npc_memory",
            "description": (
                "Returns the NPC's personal observation log — ONLY items the NPC has personally "
                "witnessed during its travels. This is embodied knowledge; it never includes "
                "objects the NPC has not seen. Use when the player asks what the NPC knows. "
                "Includes natural-language context lines ready for prompt injection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_id": {
                        "type": "string",
                        "description": "The NPC identifier.",
                    },
                    "filter_name": {
                        "type": "string",
                        "description": (
                            "Optional. Restrict results to a single object type "
                            "by Skyrim name, e.g. 'crimson flag'. "
                            "Omit to retrieve all observations."
                        ),
                    },
                },
                "required": ["npc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exploration_status",
            "description": (
                "Returns what fraction of the world the NPC has explored and which quadrant "
                "regions (NW, NE, SW, SE) it has visited. Use when the player asks whether "
                "the NPC has been to a specific area, or to explain gaps in the NPC's knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_id": {
                        "type": "string",
                        "description": "The NPC identifier.",
                    }
                },
                "required": ["npc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_object_at",
            "description": (
                "Returns the object at a specific grid position, or null if the cell is empty. "
                "Use when the NPC remembers seeing something at an exact spot and needs to "
                "confirm whether it is still there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Grid X coordinate."},
                    "y": {"type": "integer", "description": "Grid Y coordinate."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_objects",
            "description": (
                "Returns every uncollected object in the world regardless of NPC observations. "
                "This is PERFECT-KNOWLEDGE mode. Only call this when running the omniscient "
                "NPC baseline experiment condition. Do NOT use for embodied NPC responses."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_npc_target",
            "description": (
                "Commands the NPC to autonomously navigate to a specific grid position. "
                "The NPC will pathfind toward (x, y), moving one step per game tick, and "
                "will resume its default behaviour once it arrives. "
                "Use this to make the NPC physically move — for example, to lead the player "
                "to a known object, investigate a location, or reposition itself before "
                "continuing a conversation. Only call this when deliberate NPC movement is "
                "appropriate; do not call it simply to report information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_id": {
                        "type": "string",
                        "description": "The NPC identifier, e.g. 'npc_0'.",
                    },
                    "x": {
                        "type": "integer",
                        "description": "Target grid X coordinate (0 to world_size - 1).",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Target grid Y coordinate (0 to world_size - 1).",
                    },
                },
                "required": ["npc_id", "x", "y"],
            },
        },
    },
]


EMBODIED_TOOL_NAMES = {
    "get_world_info",
    "get_npc_state",
    "get_player_state",
    "get_npc_memory",
    "get_exploration_status",
    "set_npc_target",
}


def get_tool_schemas_for_knowledge_mode(is_embodied: bool) -> list[dict[str, Any]]:
    """
    Return tool schemas for the experiment's knowledge condition.

    Embodied NPCs only receive tools that expose their own state, the player
    state, quest metadata, and their personal observation memory. Full-world
    lookup tools stay available only for perfect-knowledge baselines.
    """
    if not is_embodied:
        return list(GAME_TOOL_SCHEMAS)
    return [
        schema
        for schema in GAME_TOOL_SCHEMAS
        if schema["function"]["name"] in EMBODIED_TOOL_NAMES
    ]


# ── Tool call dispatcher ───────────────────────────────────────────────────────

def get_natural_position_name(x: int, y: int, world_size: int) -> str:
    """
    Convert a grid position into a human-readable location description.

    Divides the world into a 3×3 region grid (corners, edges, centre) and
    adds a proximity qualifier ("deep in", "near the edge of") based on how
    close the position is to the nearest border.

    Examples
    ────────
        get_natural_position_name(1, 1, 20)   → "the far northwest corner"
        get_natural_position_name(10, 10, 20) → "the centre"
        get_natural_position_name(10, 2, 20)  → "the northern edge"
        get_natural_position_name(3, 14, 20)  → "the western side, toward the south"
    """
    # Normalise to [0.0, 1.0]
    fx = x / max(world_size - 1, 1)
    fy = y / max(world_size - 1, 1)

    # Horizontal band
    if fx < 0.33:
        h_band = "west"
    elif fx > 0.66:
        h_band = "east"
    else:
        h_band = "centre"

    # Vertical band  (y=0 is top/north in grid coords)
    if fy < 0.33:
        v_band = "north"
    elif fy > 0.66:
        v_band = "south"
    else:
        v_band = "centre"

    # Proximity to the nearest border (0 = at edge, 0.5 = dead centre)
    border_proximity = min(fx, 1 - fx, fy, 1 - fy)
    if border_proximity < 0.08:
        qualifier = "the far "
    elif border_proximity < 0.2:
        qualifier = "the "
    else:
        qualifier = "the "   # dropped for centre region below

    # Compose the description
    if h_band == "centre" and v_band == "centre":
        if border_proximity >= 0.2:
            return "the centre"
        return "near the centre"

    if h_band == "centre":
        edge = "northern" if v_band == "north" else "southern"
        return f"{qualifier}{edge} edge"

    if v_band == "centre":
        edge = "western" if h_band == "west" else "eastern"
        return f"{qualifier}{edge} side"

    # Corner
    direction = f"{v_band}{h_band}"   # e.g. "northwest"
    far = border_proximity < 0.08
    return f"{'the far ' if far else 'the '}{direction} corner"

def dispatch_tool_call(
    api: GameAPIProvider,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a single LLM tool call against a GameAPIProvider and return the
    result as a plain dict (ready for a provider-native tool response).

    Raises ValueError for unknown tool names so the caller can feed an error
    message back to the model rather than crashing.

    Example usage inside a provider loop
    ────────────────────────────────────
        result = dispatch_tool_call(api, function_call.name, function_call.args)
    """
    dispatch: dict[str, Any] = {
        "get_world_info":        lambda: api.get_world_info(),
        "get_npc_state":         lambda: api.get_npc_state(arguments["npc_id"]),
        "get_player_state":      lambda: api.get_player_state(),
        "get_nearby_objects":    lambda: api.get_nearby_objects(
                                     arguments["x"],
                                     arguments["y"],
                                     arguments.get("radius", 3),
                                 ),
        "get_npc_memory":        lambda: api.get_npc_memory(
                                     arguments["npc_id"],
                                     arguments.get("filter_name"),
                                 ),
        "get_exploration_status": lambda: api.get_exploration_status(arguments["npc_id"]),
        "get_object_at":         lambda: api.get_object_at(arguments["x"], arguments["y"]),
        "get_all_objects":       lambda: api.get_all_objects(),
        "set_npc_target":        lambda: api.set_npc_target(
                                     arguments["npc_id"],
                                     arguments["x"],
                                     arguments["y"],
                                 ),
    }

    if tool_name not in dispatch:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {list(dispatch)}"
        )

    return dispatch[tool_name]()
