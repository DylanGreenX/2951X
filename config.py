"""All tunable constants for the toy demo."""

# Game
PLAY_MODE = True

# Grid
GRID_SIZE = 40
CELL_PX = 18

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

# Sprite assets overlaid on the painted map. Missing files fall back to the
# colored-polygon draw path in main._draw_grid.
BG_IMAGE_PATH = "images/bg.png"
SHAPE_ASSETS = {
    "blue_circle":   "images/soul_gem.png",
    "red_circle":    "images/gem_crimson.png",
    "green_circle":  "images/gem_emerald.png",
    "purple_circle": "images/gem_amethyst.png",
    "yellow_circle": "images/gem_golden.png",

    "red_triangle":    "images/flag_crimson.png",
    "blue_triangle":   "images/flag_sapphire.png",
    "green_triangle":  "images/flag_emerald.png",
    "purple_triangle": "images/flag_amethyst.png",
    "yellow_triangle": "images/flag_golden.png",

    "green_square":  "images/alchemy_ingredient.png",
    "red_square":    "images/rune_crimson.png",
    "blue_square":   "images/rune_sapphire.png",
    "purple_square": "images/rune_amethyst.png",
    "yellow_square": "images/rune_golden.png",
}

# Fixed shape placements lined up with painted regions. Used only when
# RANDOM_SPAWN is False; otherwise world.py falls back to random placement.
# Coverage need not be complete — any shape without a fixed position gets a
# random free cell.
RANDOM_SPAWN = True
FIXED_SHAPE_POSITIONS = {
    "red_triangle":    (5, 5),
    "blue_circle":     (2, 8),
    "green_square":    (12, 3),
    "yellow_circle":   (7, 12),
    "purple_triangle": (10, 10),
    "blue_square":     (1, 14),
    "red_circle":      (14, 1),
    "green_circle":    (4, 2),
    "purple_circle":   (8, 4),
    "yellow_square":   (3, 11),
    "blue_triangle":   (11, 6),
    "green_triangle":  (6, 9),
    "purple_square":   (9, 2),
    "yellow_triangle": (2, 2),
    "red_square":      (13, 12),
}

# NPC
NPC_COUNT = 4
# Optional explicit spawn list for multi-NPC play. When None, startup picks
# stable default positions around the map and avoids the player start cell.
NPC_STARTS: list[tuple[int, int]] | None = None
# "shared"      → NPCs instantly share their observed memory
# "independent" → NPCs keep separate memory unless they exchange it locally
NPC_MULTIPLE_KNOWLEDGE_MODE = "independent"  # "shared" | "independent"
# When True and knowledge mode is "independent", NPCs deterministically
# exchange memory with other NPCs they can currently see. These exchanges are
# logged but never surfaced in the play-mode UI.
NPC_NPC_INTERACTION_ENABLED = True
NPC_SIGHT_RANGE = 8          # observes (2*r+1)x(2*r+1)
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
NPC_USE_LLM_JUDGE = False
# Judge must differ from the agent model so dual-logged agreement is not
# self-grading. Agent is now gemini-2.5-flash, so judge moves up to 2.5-pro.
NPC_JUDGE_MODEL = "gemini-2.5-pro"
NPC_OBSERVED_CELLS_VISIBLE = False # when False, NPC sight range and observed cells are not visible to player
# "deterministic" | "llm" | "slm"
NPC_RESPONSE_MODE = "slm"
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

# Local Hugging Face SLM backend. Change NPC_SLM_MODEL_ID to the "1.7b"
# preset to test the larger SmolLM checkpoint without changing code.
NPC_SLM_MODEL_PRESETS = {
    "135m": "HuggingFaceTB/SmolLM-135M",
    "1.7b": "HuggingFaceTB/SmolLM-1.7B",
    "1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "360m-instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
}
NPC_SLM_MODEL_ID = NPC_SLM_MODEL_PRESETS["1.7b-instruct"]
NPC_SLM_DEVICE = "auto"          # "auto" | "cuda" | "mps" | "cpu"
NPC_SLM_DTYPE = "auto"           # "auto" | "float16" | "bfloat16" | "float32"
NPC_SLM_MAX_NEW_TOKENS = 192
NPC_SLM_DO_SAMPLE = False
NPC_SLM_TEMPERATURE = 0.2
NPC_SLM_TOP_P = 0.9
NPC_SLM_ENABLE_TOOL_CALLS = True
NPC_SLM_MAX_TOOL_TURNS = 3
# Wrap the SLM prompt in the tokenizer's chat template before generation.
# Must be True for instruct/chat checkpoints (SmolLM2-*-Instruct, Qwen, etc)
# or instruct tuning never activates. Leave False for base completion models
# (SmolLM-1.7B, SmolLM-135M) which do not ship a chat template.
NPC_SLM_USE_CHAT_TEMPLATE = True
# When True AND tool calls are enabled, append grid-coordinate hints to the
# SLM's Known facts for each observed shape. Lets the SLM call set_npc_target
# directly without first routing through a get_npc_memory lookup (which a
# 1.7B model struggles to chain in 2 turns). Independent of the LLM path.
NPC_SLM_INCLUDE_COORDS = True
# Narrow the SLM tool menu to a curated subset. A 1.7B model confuses similar
# tools (`get_npc_state` vs `set_npc_target`) when presented with the full
# menu. Pruning to the action-as-answer tool plus memory lookup drives the
# selection rate sharply higher. Set to None to keep the full menu (same as
# the LLM path).
NPC_SLM_TOOL_WHITELIST: list[str] | None = ["set_npc_target"]
# Post-process SLM final strings: if the model mentions a region that is NOT
# among the NPC's observed regions, rewrite the response to a pure refusal.
# Mitigates the "leak-and-negate" failure mode ("I have not seen the flag in
# the merchant quarter") produced by instruct models' commit-bias. Independent
# of NPC_ENFORCE_GROUNDING which only catches literal (x, y) claims.
NPC_SLM_REGION_GROUNDING = False

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
# Multi-NPC goal assignment. When None, NPC_COMPETING keeps its old meaning:
#   True  → all NPCs compete for the player's target
#   False → no NPCs compete
# When int, exactly that many NPCs receive the player's target as their goal.
NPC_COMPETING_COUNT: int | None = None
# How non-competing NPC goals are distributed in multi-NPC play.
# "shared" → all non-competing NPCs use the same alternative goal
# "unique" → each non-competing NPC receives a distinct alternative goal
NPC_NONCOMPETING_GOAL_MODE = "shared"  # "shared" | "unique"

# Player
PLAYER_START = (13, 13)
PLAYER_SIGHT_RANGE = 20

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
# Fog is near-opaque over the painted map so unexplored regions stay hidden;
# seen tint is a subtle dim so previously-observed cells read as "memory".
FOG_COLOR = (10, 10, 15, 235)
SEEN_TINT = (10, 10, 15, 110)
NPC_COLOR = (255, 200, 50)
PLAYER_COLOR = (255, 255, 255)
TEXT_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (100, 200, 255)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # paid tier; reliable text output
# Note: gemini-2.5-flash-lite is cheaper but returns 0 output tokens on ~15%
# of trials after one tool call, leaving response_text empty. Flash doesn't.
