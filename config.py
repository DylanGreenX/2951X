"""All tunable constants for the toy demo."""

# Grid
GRID_SIZE = 15
CELL_PX = 40

# Shapes scattered in the world
NUM_BLUE_CIRCLES = 5
NUM_RED_TRIANGLES = 3
NUM_GREEN_SQUARES = 4
NUM_YELLOW_TRIANGLES = 3

# NPC
NPC_SIGHT_RANGE = 2          # observes (2*r+1)x(2*r+1) = 5x5 window
NPC_START = (1, 1)
NPC_TICK_INTERVAL = 200      # ms between NPC steps

# Player (static in this demo — we just watch the NPC)
PLAYER_START = (13, 13)

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
