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
NATURAL_LOCATIONS = {}

SHAPE_ASSETS = {
    "blue_circle":   f"images/soul_gem.png",
    "red_circle":    f"images/gem_crimson.png",
    "green_circle":  f"images/gem_emerald.png",
    "purple_circle": f"images/gem_amethyst.png",
    "yellow_circle": f"images/gem_golden.png",

    "red_triangle":    f"images/flag_crimson.png",
    "blue_triangle":   f"images/flag_sapphire.png",
    "green_triangle":  f"images/flag_emerald.png",
    "purple_triangle": f"images/flag_amethyst.png",
    "yellow_triangle": f"images/flag_golden.png",

    "green_square":  f"images/alchemy_ingredient.png",
    "red_square":    f"images/rune_crimson.png",
    "blue_square":   f"images/rune_sapphire.png",
    "purple_square": f"images/rune_amethyst.png",
    "yellow_square": f"images/rune_golden.png",
}

for x in range(15):
    for y in range(15):
        # --- TOWN REGION (Top Left) ---
        if x < 5 and y < 6:
            if x < 3 and y < 3:
                loc = "the tavern square"
            elif x >= 3 and y < 3:
                loc = "near the north gate"
            else:
                loc = "the merchant quarter"
        
        # --- FARM & WINDMILL REGION (Top Right) ---
        elif x >= 10 and y < 5:
            if x > 12:
                loc = "the sheep pastures"
            else:
                loc = "near the windmill"

        # --- THE RIVER (Central Vertical Strip) ---
        elif 6 <= x <= 8 and y < 11:
            if y == 4 or y == 5:
                loc = "the river bridge"
            else:
                loc = "the riverside"

        # --- THE SWAMP (Bottom Left) ---
        elif x < 7 and y >= 9:
            if y < 11:
                loc = "the north swamp edge"
            elif y > 13:
                loc = "the south swamp edge"
            else:
                loc = "the deep swamp"

        # --- MOUNTAINS & VOLCANO (Bottom Right) ---
        elif x >= 10 and y >= 8:
            if x >= 13 and y >= 13:
                loc = "the dragon's lair"
            elif 10 <= x <= 12 and 10 <= y <= 12:
                loc = "the volcanic crater"
            else:
                loc = "the mountain peaks"

        # --- RUINS & FOREST (Center/Eastern areas) ---
        elif x >= 9 and 5 <= y <= 7:
            loc = "the ancient stone circle"
        elif x < 10 and y >= 6:
            loc = "the dark forest"
            
        # --- DEFAULT FALLBACK ---
        else:
            loc = "the wilderness"

        NATURAL_LOCATIONS[(x, y)] = loc

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
FOG_COLOR = (10, 10, 15, 255)
SEEN_TINT = (10, 10, 15, 100)
NPC_COLOR = (255, 200, 50)
PLAYER_COLOR = (255, 255, 255)
TEXT_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (100, 200, 255)
BG_IMAGE_PATH = "images/bg.png"