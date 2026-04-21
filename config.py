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
# Location naming is region-based — see game_api_interface.get_natural_position_name.
# The old flat per-cell alias map was replaced because arbitrary landmark names
# ("the tavern", "the blacksmith") carried no geographic meaning and forced the
# LLM to reconcile three representations (cell coord, shape label, landmark).

# NPC
NPC_SIGHT_RANGE = 1          # observes (2*r+1)x(2*r+1) = 3x3 window
NPC_START = (1, 1)
NPC_TICK_INTERVAL = 600      # ms between NPC steps
NPC_EXPLORATION_TICKS = 40   # ticks the brain runs before being asked (experiment)

# Knowledge-acquisition modalities — applied inside RLangState.observe().
# Composable with each other and with NPC_COMPETING.
#
# NPC_MEMORY_DECAY_TICKS: if int, shapes first seen more than N ticks ago
#   are culled from memory every observation step. None disables.
# NPC_SELECTIVE_ATTENTION: "color" matches NPC_GOAL_COLOR, "shape" matches
#   NPC_GOAL_SHAPE; non-matching shapes are ignored at observation time.
#   None disables.
NPC_MEMORY_DECAY_TICKS: int | None = None
NPC_SELECTIVE_ATTENTION: str | None = None  # "color" | "shape" | None

# Response scoring — when True, the experiment runs an LLM-based judge in
# addition to the regex metrics and dual-logs both. Judge model defaults to
# gemini-2.5-flash (different tier than the agent for independence).
NPC_USE_LLM_JUDGE = True
# Judge must differ from the agent model so dual-logged agreement is not
# self-grading. Agent is now gemini-2.5-flash, so judge moves up to 2.5-pro.
NPC_JUDGE_MODEL = "gemini-2.5-pro"
NPC_OBSERVED_CELLS_VISIBLE = False # when False, NPC sight range and observed cells are not visible to player
# "deterministic" | "llm" | "slm"
NPC_RESPONSE_MODE = "llm"
NPC_LLM_MAX_TOOL_TURNS = 4
NPC_LLM_MAX_OUTPUT_TOKENS = 128
NPC_LLM_TEMPERATURE = 0.4
NPC_LLM_TIMEOUT_MS = 30000
NPC_ENFORCE_GROUNDING = PLAY_MODE
NPC_LLM_LOG_ENABLED = True
# Fallback LLM event log path. GameLogger retargets this per run so LLM
# events flow into the run's game.jsonl; this value is only used if a run
# log has not been started (e.g. during unit tests or ad-hoc scripts).
NPC_LLM_LOG_PATH = "llm_interactions.jsonl"
# Root directory for per-run game logs. Each run creates a subdirectory
# here containing game.jsonl (events) and summary.json (outcome).
GAME_LOG_DIR = "logs/runs"

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

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # paid tier; reliable text output
# Note: gemini-2.5-flash-lite is cheaper but returns 0 output tokens on ~15%
# of trials after one tool call, leaving response_text empty. Flash doesn't.