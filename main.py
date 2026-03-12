"""
Toy demo: NPC with RLang-grounded knowledge.

The NPC pursues its own goal (collecting blue circles) and
accumulates incidental observations about the world. The bottom
panel shows the RLang state serialized as natural language strings —
exactly what would be injected into an LLM prompt.

Controls:
  Arrow keys — move player
  R — reset world
  ESC / close — quit
"""
import sys
import math
import pygame
import config
from world import GameWorld
from entities import Player, NPC
from npc_brain import NPCBrain


# ── Color palette for shapes ──────────────────────────────────
SHAPE_COLORS = {
    "red":    (220, 60, 60),
    "blue":   (60, 120, 230),
    "green":  (60, 200, 90),
    "yellow": (230, 210, 50),
}


def main():
    pygame.init()

    grid_px = config.GRID_SIZE * config.CELL_PX
    win_w = grid_px + config.SIDEBAR_W
    win_h = config.TOP_BAR_H + grid_px + config.HUD_H
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("RLang NPC Demo — Needle in a Haystack")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", config.FONT_SIZE)
    font_big = pygame.font.SysFont("consolas", 18, bold=True)
    font_title = pygame.font.SysFont("consolas", 14, bold=True)

    # ── Init world ──
    world = GameWorld(seed=42)
    player = Player(*config.PLAYER_START)
    npc = NPC(*config.NPC_START, sight_range=config.NPC_SIGHT_RANGE)
    brain = NPCBrain(npc, world)

    # Event log (notable things that happened)
    event_log: list[str] = []

    last_npc_tick = pygame.time.get_ticks()
    running = True

    while running:
        now = pygame.time.get_ticks()

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    world = GameWorld(seed=None)
                    player = Player(*config.PLAYER_START)
                    npc = NPC(*config.NPC_START, sight_range=config.NPC_SIGHT_RANGE)
                    brain = NPCBrain(npc, world)
                    event_log.clear()
                # Player movement
                dx, dy = 0, 0
                if event.key == pygame.K_UP: dy = -1
                elif event.key == pygame.K_DOWN: dy = 1
                elif event.key == pygame.K_LEFT: dx = -1
                elif event.key == pygame.K_RIGHT: dx = 1
                nx, ny = player.x + dx, player.y + dy
                if world.in_bounds(nx, ny):
                    player.x, player.y = nx, ny

        # ── NPC tick ──
        if now - last_npc_tick >= config.NPC_TICK_INTERVAL:
            result = brain.tick()
            if result:
                event_log.append(result)
                if len(event_log) > 8:
                    event_log.pop(0)
            last_npc_tick = now

        # ── Draw ──
        screen.fill(config.BG_COLOR)
        _draw_top_bar(screen, font_big, npc, brain, world)
        _draw_grid(screen, world, brain, npc, player, font)
        _draw_sidebar(screen, font_title, font, brain, event_log, grid_px)
        _draw_hud(screen, font_title, font, brain, grid_px, win_w)

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


# ── Drawing helpers ────────────────────────────────────────────

def _grid_origin():
    return (0, config.TOP_BAR_H)


def _draw_top_bar(screen, font, npc, brain, world):
    """Status bar at top."""
    bar = pygame.Rect(0, 0, screen.get_width(), config.TOP_BAR_H)
    pygame.draw.rect(screen, (40, 40, 50), bar)

    coverage = brain.state.coverage
    txt = (
        f"NPC Steps: {npc.steps_taken}    "
        f"Blue Circles Collected: {npc.blue_circles_collected}/{config.NUM_BLUE_CIRCLES}    "
        f"World Explored: {coverage:.0%}    "
        f"Shapes Seen: {len(brain.state.observed_shapes)}"
    )
    surf = font.render(txt, True, config.TEXT_COLOR)
    screen.blit(surf, (12, 15))


def _draw_grid(screen, world, brain, npc, player, font):
    """Draw the grid, shapes, fog of war, NPC, and player."""
    ox, oy = _grid_origin()

    # Grid lines
    for i in range(config.GRID_SIZE + 1):
        px = ox + i * config.CELL_PX
        pygame.draw.line(screen, config.GRID_LINE_COLOR, (px, oy), (px, oy + config.GRID_SIZE * config.CELL_PX))
        py = oy + i * config.CELL_PX
        pygame.draw.line(screen, config.GRID_LINE_COLOR, (ox, py), (ox + config.GRID_SIZE * config.CELL_PX, py))

    # Fog of war + seen tint
    fog_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
    fog_surf.fill(config.FOG_COLOR)
    seen_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
    seen_surf.fill(config.SEEN_TINT)

    for gx in range(config.GRID_SIZE):
        for gy in range(config.GRID_SIZE):
            px = ox + gx * config.CELL_PX
            py = oy + gy * config.CELL_PX
            if (gx, gy) in brain.state.observed_cells:
                screen.blit(seen_surf, (px, py))
            else:
                screen.blit(fog_surf, (px, py))

    # NPC sight range highlight
    sr = npc.sight_range
    sight_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
    sight_surf.fill((255, 200, 50, 25))
    for dx in range(-sr, sr + 1):
        for dy in range(-sr, sr + 1):
            gx, gy = npc.x + dx, npc.y + dy
            if 0 <= gx < config.GRID_SIZE and 0 <= gy < config.GRID_SIZE:
                px = ox + gx * config.CELL_PX
                py = oy + gy * config.CELL_PX
                screen.blit(sight_surf, (px, py))

    # Shapes
    for shape in world.shapes:
        if shape.collected:
            continue
        px = ox + shape.x * config.CELL_PX + config.CELL_PX // 2
        py = oy + shape.y * config.CELL_PX + config.CELL_PX // 2
        color = SHAPE_COLORS[shape.color]
        r = config.CELL_PX // 3

        if shape.shape_type == "circle":
            pygame.draw.circle(screen, color, (px, py), r)
            pygame.draw.circle(screen, (255, 255, 255), (px, py), r, 2)
        elif shape.shape_type == "triangle":
            pts = [
                (px, py - r),
                (px - r, py + r),
                (px + r, py + r),
            ]
            pygame.draw.polygon(screen, color, pts)
            pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
        elif shape.shape_type == "square":
            rect = pygame.Rect(px - r, py - r, r * 2, r * 2)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)

    # NPC — diamond shape
    npx = ox + npc.x * config.CELL_PX + config.CELL_PX // 2
    npy = oy + npc.y * config.CELL_PX + config.CELL_PX // 2
    d = config.CELL_PX // 3
    npc_pts = [(npx, npy - d), (npx + d, npy), (npx, npy + d), (npx - d, npy)]
    pygame.draw.polygon(screen, config.NPC_COLOR, npc_pts)
    pygame.draw.polygon(screen, (255, 255, 255), npc_pts, 2)
    # Label
    label = font.render("NPC", True, config.NPC_COLOR)
    screen.blit(label, (npx - label.get_width() // 2, npy - d - 14))

    # Player — filled square with cross
    ppx = ox + player.x * config.CELL_PX + config.CELL_PX // 2
    ppy = oy + player.y * config.CELL_PX + config.CELL_PX // 2
    pr = config.CELL_PX // 3
    prect = pygame.Rect(ppx - pr, ppy - pr, pr * 2, pr * 2)
    pygame.draw.rect(screen, config.PLAYER_COLOR, prect)
    pygame.draw.rect(screen, (100, 100, 100), prect, 2)
    # Cross marker
    pygame.draw.line(screen, (0, 0, 0), (ppx - pr//2, ppy), (ppx + pr//2, ppy), 2)
    pygame.draw.line(screen, (0, 0, 0), (ppx, ppy - pr//2), (ppx, ppy + pr//2), 2)
    label = font.render("YOU", True, config.PLAYER_COLOR)
    screen.blit(label, (ppx - label.get_width() // 2, ppy - pr - 14))


def _draw_sidebar(screen, font_title, font, brain, event_log, grid_px):
    """Right sidebar: legend + event log."""
    ox = config.GRID_SIZE * config.CELL_PX + 10
    oy = config.TOP_BAR_H + 10

    # Legend
    title = font_title.render("LEGEND", True, config.HIGHLIGHT_COLOR)
    screen.blit(title, (ox, oy))
    oy += 24

    legend_items = [
        ("NPC (yellow diamond)", config.NPC_COLOR),
        ("Player (white square)", config.PLAYER_COLOR),
        ("Blue circle = NPC's goal", SHAPE_COLORS["blue"]),
        ("Red triangle = YOUR goal", SHAPE_COLORS["red"]),
        ("Green square = neutral", SHAPE_COLORS["green"]),
        ("Yellow triangle = neutral", SHAPE_COLORS["yellow"]),
        ("Bright cells = NPC explored", (80, 80, 90)),
        ("Dark cells = unexplored (fog)", (40, 40, 45)),
    ]

    for text, color in legend_items:
        pygame.draw.rect(screen, color, (ox, oy + 2, 12, 12))
        surf = font.render(text, True, config.TEXT_COLOR)
        screen.blit(surf, (ox + 18, oy))
        oy += 20

    # Event log
    oy += 16
    title = font_title.render("NPC EVENT LOG", True, config.HIGHLIGHT_COLOR)
    screen.blit(title, (ox, oy))
    oy += 22

    for entry in event_log[-8:]:
        surf = font.render(f"> {entry}", True, (180, 220, 180))
        screen.blit(surf, (ox, oy))
        oy += 18


def _draw_hud(screen, font_title, font, brain, grid_px, win_w):
    """Bottom panel: RLang state serialized as LLM context strings."""
    oy = config.TOP_BAR_H + grid_px + 4

    # Background
    hud_rect = pygame.Rect(0, oy, win_w, config.HUD_H)
    pygame.draw.rect(screen, (25, 25, 35), hud_rect)
    pygame.draw.line(screen, config.HIGHLIGHT_COLOR, (0, oy), (win_w, oy), 2)

    oy += 6
    title = font_title.render("  LLM CONTEXT (RLang State -> Natural Language)", True, config.HIGHLIGHT_COLOR)
    screen.blit(title, (4, oy))
    oy += 22

    # Get the serialized RLang state
    context_lines = brain.state.to_llm_context()

    for line in context_lines:
        # Color code by tag
        if "[GOAL]" in line:
            color = SHAPE_COLORS["blue"]
        elif "[PROPOSITION]" in line and "red triangle" in line.lower():
            color = SHAPE_COLORS["red"]
        elif "[PROPOSITION]" in line:
            color = (180, 180, 220)
        elif "[OBSERVATION]" in line:
            color = (180, 220, 180)
        elif "[FACTOR]" in line:
            color = (220, 200, 150)
        else:
            color = config.TEXT_COLOR

        # Truncate long lines
        display = line if len(line) < 90 else line[:87] + "..."
        surf = font.render(display, True, color)
        screen.blit(surf, (12, oy))
        oy += 17

        if oy > config.TOP_BAR_H + grid_px + config.HUD_H - 10:
            surf = font.render("  ... (more)", True, (120, 120, 120))
            screen.blit(surf, (12, oy))
            break


if __name__ == "__main__":
    main()
