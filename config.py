"""All tunable constants for the toy demo."""

# Game
PLAY_MODE = True

# Grid
GRID_SIZE = 15
CELL_PX = 40

# Shapes scattered in the world
COLORS = ["blue", "green", "purple", "red", "yellow"]
SHAPES = ["circle", "triangle", "square"]
NUM_OBJECTS = 15

# Skyrim vocabulary — maps internal labels to natural language for LLM prompts
NATURAL_COLORS = {
    "red": "crimson",
    "blue": "sapphire",
    "green": "emerald",
    "yellow": "golden",
    "purple": "amethyst"
}
NATURAL_SHAPES = {
    "triangle": "banner",
    "circle": "gem",
    "square": "rune stone"
}
# Override systematic mapping for key items
NATURAL_OBJECTS = {
    "red_triangle": "crimson flag",
    "blue_circle": "soul gem",
    "green_square": "alchemy ingredient"
}
NATURAL_LOCATIONS = {
    (0,0): "the tavern", (0,1): "near the blacksmith", (0,2): "by the forge",
    (1,0): "market square", (1,1): "by the temple", (1,2): "temple steps",
    (2,0): "watchtower", (2,1): "outside the inn", (2,2): "inn courtyard",
    (3,0): "alchemy shop", (3,1): "near the stables", (3,2): "stable yard",
    (4,0): "guard barracks", (4,1): "by the well", (4,2): "town center",
    (5,0): "city gate", (5,1): "near the walls", (5,2): "outer district",
    (6,0): "main road", (6,1): "crossroads", (6,2): "merchant quarter",
    (7,0): "old bridge", (7,1): "riverside", (7,2): "fishing spot",
    (8,0): "windmill", (8,1): "grain fields", (8,2): "farmhouse",
    (9,0): "ancient ruins", (9,1): "stone circle", (9,2): "burial ground",
    (10,0): "forest edge", (10,1): "hunting grounds", (10,2): "ranger camp",
    (11,0): "mountain path", (11,1): "rocky outcrop", (11,2): "cave entrance",
    (12,0): "northern gate", (12,1): "guard post", (12,2): "watchtower base",
    (13,0): "eastern border", (13,1): "trade route", (13,2): "caravan stop",
    (14,0): "southern outpost", (14,1): "frontier settlement", (14,2): "wilderness edge"
}

# NPC
NPC_SIGHT_RANGE = 2          # observes (2*r+1)x(2*r+1) = 5x5 window
NPC_START = (1, 1)
NPC_TICK_INTERVAL = 600      # ms between NPC steps
NPC_OBSERVED_CELLS_VISIBLE = False # when False, NPC sight range and observed cells are not visible to player
# "deterministic" | "llm" | "slm"
NPC_RESPONSE_MODE = "llm"
NPC_LLM_MAX_TOOL_TURNS = 4
NPC_LLM_MAX_OUTPUT_TOKENS = 128
NPC_LLM_TEMPERATURE = 0.4
NPC_LLM_TIMEOUT_MS = 30000
NPC_ENFORCE_GROUNDING = PLAY_MODE
NPC_LLM_LOG_ENABLED = True
NPC_LLM_LOG_PATH = "llm_interactions.jsonl"

# NPC Knowledge Mode — the knowledge axis in the experiment matrix.
# "embodied" → NPC only knows what it personally observed (realistic, default)
# "perfect"  → NPC knows the full world state (omniscient baseline)
NPC_KNOWLEDGE_MODE = "embodied"

# NPC Goal — when False, NPC uses NPCBrainWandering (baseline).
# When True, NPC uses NPCBrainGoalDriven with goal resolved as follows:
#   NPC_COMPETING=True           → goal matches the player's target
#   NPC_GOAL_DETERMINISTIC=True  → use NPC_GOAL_COLOR + NPC_GOAL_SHAPE
#   NPC_GOAL_DETERMINISTIC=False → random goal, guaranteed != player target
NPC_GOAL = True
NPC_GOAL_DETERMINISTIC = True
NPC_GOAL_COLOR = "blue"
NPC_GOAL_SHAPE = "circle"
NPC_COMPETING = False

# Player
PLAYER_START = (13, 13)
PLAYER_SIGHT_RANGE = 2

# Target
DETERMINISTIC_TARGET = True # when false, target is randomly chosen from color x shape space
TARGET_COLOR = "red"
TARGET_SHAPE = "triangle"

# Display
SIDEBAR_W = 420
TOP_BAR_H = 50
HUD_H = 220                  # bottom panel for LLM context strings
FONT_SIZE = 14
FPS = 30

# Colors
BG_COLOR = (30, 30, 35)
GRID_LINE_COLOR = (50, 50, 55)
FOG_COLOR = (20, 20, 25, 180)
SEEN_TINT = (255, 255, 255, 18)
NPC_COLOR = (255, 200, 50)
PLAYER_COLOR = (255, 255, 255)
TEXT_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (100, 200, 255)
