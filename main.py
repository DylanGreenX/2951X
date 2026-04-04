"""
Toy demo: NPC with RLang-grounded knowledge.

The NPC wanders aimlessly to begin and
accumulates incidental observations about the world. The bottom
panel shows the RLang state serialized as natural language strings —
exactly what would be injected into an LLM prompt.

Controls:
  Arrow keys — move player
  R — reset world
  ESC / close — quit
"""
import sys
import random
import pygame
import config
from world import GameWorld
from entities import Player, NPC
from npc_brain import NPCBrainGoalDriven, NPCBrainWandering
from interaction import InteractionManager


# ── Color palette for shapes ──────────────────────────────────
SHAPE_COLORS = {
    "red":    (220, 60, 60),
    "blue":   (60, 120, 230),
    "green":  (60, 200, 90),
    "yellow": (230, 210, 50),
    "purple": (120, 80, 200),
}


def _resolve_npc_goal(target_color: str, target_shape: str) -> str | None:
    """
    Determine the NPC's goal label from config.

    Returns a shape label string (e.g. "blue_circle") or None for wandering.
    Resolution order:
      NPC_GOAL=False         → None (wandering baseline)
      NPC_COMPETING=True     → same label as the player's target
      NPC_GOAL_DETERMINISTIC → NPC_GOAL_COLOR + NPC_GOAL_SHAPE
      else                   → random label, guaranteed != player target
    """
    if not config.NPC_GOAL:
        return None
    if config.NPC_COMPETING:
        return f"{target_color}_{target_shape}"
    if config.NPC_GOAL_DETERMINISTIC:
        return f"{config.NPC_GOAL_COLOR}_{config.NPC_GOAL_SHAPE}"
    all_labels = [f"{c}_{s}" for c in config.COLORS for s in config.SHAPES]
    player_label = f"{target_color}_{target_shape}"
    choices = [lbl for lbl in all_labels if lbl != player_label]
    return random.choice(choices)


def _init_game(seed=None):
    """Create a fresh world, player, NPC, and brain. Returns (world, player, npc, brain)."""
    if config.DETERMINISTIC_TARGET:
        target_color, target_shape = config.TARGET_COLOR, config.TARGET_SHAPE
    else:
        target_color = random.choice(config.COLORS)
        target_shape = random.choice(config.SHAPES)

    world = GameWorld(target_color=target_color, target_shape=target_shape, seed=seed)

    player = Player(*config.PLAYER_START, sight_range=config.PLAYER_SIGHT_RANGE)
    world.update_player_vision(player)  # populate initial sight cone immediately

    npc = NPC(*config.NPC_START, sight_range=config.NPC_SIGHT_RANGE)

    goal_label = _resolve_npc_goal(target_color, target_shape)
    if goal_label is None:
        brain = NPCBrainWandering(npc, world)
    else:
        brain = NPCBrainGoalDriven(npc, world, goal_label=goal_label)

    interaction_manager = InteractionManager()

    return world, player, npc, brain, interaction_manager


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

    world, player, npc, brain, interaction_manager = _init_game(seed=42)

    # Event log (notable things that happened)
    event_log: list[str] = []

    # Interaction state — when in_interaction is True the game is frozen
    in_interaction: bool = False
    interaction_question: str = ""
    interaction_response: str = ""

    last_npc_tick = pygame.time.get_ticks()
    running = True

    while running:
        now = pygame.time.get_ticks()

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if in_interaction:
                    # ESC or ENTER dismisses the dialogue and resumes the game
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        in_interaction = False
                else:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        world, player, npc, brain, interaction_manager = _init_game()
                        event_log.clear()
                        in_interaction = False
                    # Player movement
                    dx, dy = 0, 0
                    if event.key == pygame.K_UP:
                        dy = -1
                    elif event.key == pygame.K_DOWN:
                        dy = 1
                    elif event.key == pygame.K_LEFT:
                        dx = -1
                    elif event.key == pygame.K_RIGHT:
                        dx = 1
                    nx, ny = player.x + dx, player.y + dy
                    if world.in_bounds(nx, ny):
                        player.x, player.y = nx, ny
                        world.update_player_vision(player)
                        # Auto-trigger interaction when player steps onto NPC
                        if interaction_manager.can_interact(player, npc):
                            interaction_question, interaction_response = (
                                interaction_manager.start_interaction(
                                    brain,
                                    world.target_color,
                                    world.target_shape,
                                )
                            )
                            in_interaction = True

        # ── NPC tick (frozen during interaction) ──
        if not in_interaction and now - last_npc_tick >= config.NPC_TICK_INTERVAL:
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
        if in_interaction:
            _draw_interaction_overlay(
                screen, font_big, font, interaction_question, interaction_response, win_w, win_h
            )

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
            if (gx, gy) in player.observed_cells:
                screen.blit(seen_surf, (px, py))
            elif (config.NPC_OBSERVED_CELLS_VISIBLE or not config.PLAY_MODE) and (gx, gy) in brain.state.observed_cells:
                screen.blit(seen_surf, (px, py))
            else:
                screen.blit(fog_surf, (px, py))

    # NPC sight range highlight
    if config.NPC_OBSERVED_CELLS_VISIBLE or not config.PLAY_MODE:
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
    
    # Player sight range highlight
    sr = player.sight_range
    sight_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
    sight_surf.fill((200, 220, 255, 100))
    for dx in range(-sr, sr + 1):
        for dy in range(-sr, sr + 1):
            gx, gy = player.x + dx, player.y + dy
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
        if (config.PLAY_MODE and (shape.x, shape.y) in player.observed_cells) or not config.PLAY_MODE or (config.NPC_OBSERVED_CELLS_VISIBLE and (shape.x, shape.y) in brain.state.observed_cells):
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
        ("Blue circle", SHAPE_COLORS["blue"]),
        ("Red triangle", SHAPE_COLORS["red"]),
        ("Green square", SHAPE_COLORS["green"]),
        ("Yellow triangle", SHAPE_COLORS["yellow"]),
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
        if "[PROPOSITION]" in line:
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


def _draw_interaction_overlay(screen, font_big, font, question: str, response: str, win_w: int, win_h: int):
    """Modal dialogue box drawn over the game when an interaction is active."""
    box_w, box_h = 540, 180
    box_x = (win_w - box_w) // 2
    box_y = (win_h - box_h) // 2

    # Dim the background
    dim = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 150))
    screen.blit(dim, (0, 0))

    # Box background + border
    pygame.draw.rect(screen, (30, 30, 45), (box_x, box_y, box_w, box_h), border_radius=8)
    pygame.draw.rect(screen, config.HIGHLIGHT_COLOR, (box_x, box_y, box_w, box_h), 2, border_radius=8)

    pad = 18
    y = box_y + pad

    # Speaker labels + text
    you_label = font_big.render("YOU:", True, config.PLAYER_COLOR)
    screen.blit(you_label, (box_x + pad, y))
    q_surf = font.render(question, True, config.TEXT_COLOR)
    screen.blit(q_surf, (box_x + pad + you_label.get_width() + 8, y + 3))
    y += you_label.get_height() + 14

    npc_label = font_big.render("NPC:", True, config.NPC_COLOR)
    screen.blit(npc_label, (box_x + pad, y))
    # Wrap response across multiple lines if needed
    max_chars = (box_w - pad * 2 - npc_label.get_width() - 8) // 8
    words = response.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    resp_x = box_x + pad + npc_label.get_width() + 8
    for line in lines:
        r_surf = font.render(line, True, (180, 220, 180))
        screen.blit(r_surf, (resp_x, y + 3))
        y += font.get_height() + 2

    # Dismiss hint
    hint = font.render("[ ENTER / ESC ]  to close", True, (120, 120, 140))
    screen.blit(hint, (box_x + box_w - hint.get_width() - pad, box_y + box_h - hint.get_height() - 10))


if __name__ == "__main__":
    main()
