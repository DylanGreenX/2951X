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
from llm import SLMClient
from pygame_game_api import PygameGameAPI
from game_log import GameLogger


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


def _build_preloaded_slm_client() -> SLMClient | None:
    """Load the local SLM once at startup so interaction stays on the hot path."""
    if config.NPC_RESPONSE_MODE != "slm":
        return None
    slm_client = SLMClient(
        model_id=getattr(config, "NPC_SLM_MODEL_ID", "HuggingFaceTB/SmolLM-135M"),
        device=getattr(config, "NPC_SLM_DEVICE", "auto"),
        dtype=getattr(config, "NPC_SLM_DTYPE", "auto"),
    )
    slm_client.preload()
    return slm_client


def _init_game(
    seed=None,
    prev_logger: GameLogger | None = None,
    slm_client: SLMClient | None = None,
):
    """Create a fresh world, player, NPC, brain, interaction manager, and game logger.

    If ``prev_logger`` is passed (e.g. on reset), it is ended first so its
    summary.json is written and ``config.NPC_LLM_LOG_PATH`` is restored
    before we start a new run.
    """
    if prev_logger is not None:
        prev_logger.end(outcome="reset")

    if config.DETERMINISTIC_TARGET:
        target_color, target_shape = config.TARGET_COLOR, config.TARGET_SHAPE
    else:
        target_color = random.choice(config.COLORS)
        target_shape = random.choice(config.SHAPES)

    world = GameWorld(target_color=target_color, target_shape=target_shape, seed=seed)

    player = Player(*config.PLAYER_START, sight_range=config.PLAYER_SIGHT_RANGE)
    world.update_player_vision(player)

    npc = NPC(*config.NPC_START, sight_range=config.NPC_SIGHT_RANGE)

    goal_label = _resolve_npc_goal(target_color, target_shape)
    if goal_label is None:
        brain = NPCBrainWandering(npc, world)
    else:
        brain = NPCBrainGoalDriven(npc, world, goal_label=goal_label)

    interaction_manager = InteractionManager(
        api=PygameGameAPI.from_game(world, player, brain),
        slm_client=slm_client,
    )

    # Start the run log. GameLogger retargets config.NPC_LLM_LOG_PATH so all
    # LLM events auto-flow into the per-run game.jsonl for auditing.
    logger = GameLogger.start(world, player, npc, brain, tag="play", seed=seed)

    return world, player, npc, brain, interaction_manager, logger


def _load_map_assets(grid_px: int) -> dict:
    """Load bg + shape + entity sprites once. Missing files silently fall
    back to None so the old colored-polygon draw path stays available."""
    assets: dict = {"bg": None, "shapes": {}, "npc": None, "player": None}

    try:
        bg = pygame.image.load(config.BG_IMAGE_PATH).convert()
        assets["bg"] = pygame.transform.scale(bg, (grid_px, grid_px))
    except (pygame.error, FileNotFoundError, AttributeError):
        pass

    sprite_px = int(config.CELL_PX * 0.75)
    for label, path in getattr(config, "SHAPE_ASSETS", {}).items():
        try:
            img = pygame.image.load(path).convert_alpha()
            assets["shapes"][label] = pygame.transform.scale(img, (sprite_px, sprite_px))
        except (pygame.error, FileNotFoundError):
            assets["shapes"][label] = None

    entity_px = int(config.CELL_PX)
    for key, path in (("npc", "images/npc.png"), ("player", "images/player.png")):
        try:
            img = pygame.image.load(path).convert_alpha()
            assets[key] = pygame.transform.scale(img, (entity_px, entity_px))
        except (pygame.error, FileNotFoundError):
            pass

    return assets


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

    assets = _load_map_assets(grid_px)

    shared_slm_client = _build_preloaded_slm_client()
    world, player, npc, brain, interaction_manager, logger = _init_game(
        seed=42,
        slm_client=shared_slm_client,
    )

    # Event log (notable things that happened)
    event_log: list[str] = []

    # Interaction state — when in_interaction is True the game is frozen
    in_interaction: bool = False
    interaction_question: str = ""
    interaction_response: str = ""

    last_npc_tick = pygame.time.get_ticks()
    running = True
    outcome = "quit"

    while running:
        now = pygame.time.get_ticks()

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if in_interaction:
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        in_interaction = False
                else:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        brain.set_target_pos((player.x,player.y))
                    elif event.key == pygame.K_r:
                        world, player, npc, brain, interaction_manager, logger = (
                            _init_game(
                                prev_logger=logger,
                                slm_client=shared_slm_client,
                            )
                        )
                        event_log.clear()
                        in_interaction = False
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
                        logger.log_tick(world, player, npc, brain)
                        # Auto-trigger interaction when player steps onto NPC
                        if interaction_manager.can_interact(player, npc):
                            interaction_id = logger.log_interaction_pre(
                                world, player, npc, brain,
                                world.target_color, world.target_shape,
                            )
                            interaction_question, interaction_response = (
                                interaction_manager.start_interaction(
                                    brain,
                                    world.target_color,
                                    world.target_shape,
                                )
                            )
                            logger.log_interaction_summary(
                                interaction_id, interaction_manager, brain, world,
                                world.target_color, world.target_shape,
                                interaction_question, interaction_response,
                            )
                            in_interaction = True

        # ── NPC tick (frozen during interaction) ──
        if not in_interaction and now - last_npc_tick >= config.NPC_TICK_INTERVAL:
            result = brain.tick()
            if result:
                event_log.append(result)
                if len(event_log) > 8:
                    event_log.pop(0)
            logger.log_tick(world, player, npc, brain, event_msg=result)
            last_npc_tick = now

        # ── Draw ──
        screen.fill(config.BG_COLOR)
        _draw_top_bar(screen, font_big, npc, brain, world)
        _draw_grid(screen, world, brain, npc, player, font, assets)
        _draw_sidebar(screen, font_title, font, brain, event_log, grid_px)
        _draw_hud(screen, font_title, font, brain, grid_px, win_w)
        if in_interaction:
            _draw_interaction_overlay(
                screen, font_big, font, interaction_question, interaction_response, win_w, win_h
            )

        pygame.display.flip()
        clock.tick(config.FPS)

    logger.end(
        outcome=outcome,
        extra_stats={
            "npc_steps": npc.steps_taken,
            "npc_coverage": round(brain.state.coverage, 4),
            "player_pos": [player.x, player.y],
            "npc_pos": [npc.x, npc.y],
        },
    )
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


def _draw_grid(screen, world, brain, npc, player, font, assets):
    """Draw the painted map, shapes, fog of war, NPC, and player.

    Render order: bg → grid lines → shape sprites → entities → fog overlay.
    Fog is applied last so it tints bg + sprites uniformly.
    """
    ox, oy = _grid_origin()
    grid_px = config.GRID_SIZE * config.CELL_PX

    # 1. Painted map (fallback: solid background fill is already on-screen)
    if assets.get("bg") is not None:
        screen.blit(assets["bg"], (ox, oy))

    # 2. Grid lines on top of map for cell readability
    for i in range(config.GRID_SIZE + 1):
        px = ox + i * config.CELL_PX
        pygame.draw.line(screen, config.GRID_LINE_COLOR, (px, oy), (px, oy + grid_px))
        py = oy + i * config.CELL_PX
        pygame.draw.line(screen, config.GRID_LINE_COLOR, (ox, py), (ox + grid_px, py))

    # 3. Shape sprites (or polygon fallback) for anything the player has seen
    shape_sprites = assets.get("shapes", {})
    for shape in world.shapes:
        if shape.collected:
            continue
        visible = (
            (config.PLAY_MODE and (shape.x, shape.y) in player.observed_cells)
            or not config.PLAY_MODE
            or (config.NPC_OBSERVED_CELLS_VISIBLE and (shape.x, shape.y) in brain.state.observed_cells)
        )
        if not visible:
            continue

        label = f"{shape.color}_{shape.shape_type}"
        sprite = shape_sprites.get(label)
        if sprite is not None:
            ix = ox + shape.x * config.CELL_PX + (config.CELL_PX - sprite.get_width()) // 2
            iy = oy + shape.y * config.CELL_PX + (config.CELL_PX - sprite.get_height()) // 2
            screen.blit(sprite, (ix, iy))
        else:
            px = ox + shape.x * config.CELL_PX + config.CELL_PX // 2
            py = oy + shape.y * config.CELL_PX + config.CELL_PX // 2
            color = SHAPE_COLORS[shape.color]
            r = config.CELL_PX // 3
            if shape.shape_type == "circle":
                pygame.draw.circle(screen, color, (px, py), r)
                pygame.draw.circle(screen, (255, 255, 255), (px, py), r, 2)
            elif shape.shape_type == "triangle":
                pts = [(px, py - r), (px - r, py + r), (px + r, py + r)]
                pygame.draw.polygon(screen, color, pts)
                pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
            elif shape.shape_type == "square":
                rect = pygame.Rect(px - r, py - r, r * 2, r * 2)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (255, 255, 255), rect, 2)

    # 4. NPC
    npc_visible = (npc.x, npc.y) in player.observed_cells or not config.PLAY_MODE
    if npc_visible:
        npc_img = assets.get("npc")
        if npc_img is not None:
            ix = ox + npc.x * config.CELL_PX + (config.CELL_PX - npc_img.get_width()) // 2
            iy = oy + npc.y * config.CELL_PX + (config.CELL_PX - npc_img.get_height()) // 2
            screen.blit(npc_img, (ix, iy))
        else:
            npx = ox + npc.x * config.CELL_PX + config.CELL_PX // 2
            npy = oy + npc.y * config.CELL_PX + config.CELL_PX // 2
            d = config.CELL_PX // 3
            pts = [(npx, npy - d), (npx + d, npy), (npx, npy + d), (npx - d, npy)]
            pygame.draw.polygon(screen, config.NPC_COLOR, pts)
            pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
            label = font.render("NPC", True, config.NPC_COLOR)
            screen.blit(label, (npx - label.get_width() // 2, npy - d - 14))

    # 5. Player
    player_img = assets.get("player")
    if player_img is not None:
        ix = ox + player.x * config.CELL_PX + (config.CELL_PX - player_img.get_width()) // 2
        iy = oy + player.y * config.CELL_PX + (config.CELL_PX - player_img.get_height()) // 2
        screen.blit(player_img, (ix, iy))
    else:
        ppx = ox + player.x * config.CELL_PX + config.CELL_PX // 2
        ppy = oy + player.y * config.CELL_PX + config.CELL_PX // 2
        pr = config.CELL_PX // 3
        prect = pygame.Rect(ppx - pr, ppy - pr, pr * 2, pr * 2)
        pygame.draw.rect(screen, config.PLAYER_COLOR, prect)
        pygame.draw.rect(screen, (100, 100, 100), prect, 2)
        pygame.draw.line(screen, (0, 0, 0), (ppx - pr // 2, ppy), (ppx + pr // 2, ppy), 2)
        pygame.draw.line(screen, (0, 0, 0), (ppx, ppy - pr // 2), (ppx, ppy + pr // 2), 2)
        label = font.render("YOU", True, config.PLAYER_COLOR)
        screen.blit(label, (ppx - label.get_width() // 2, ppy - pr - 14))

    # 6. Fog overlay — applied last so bg + sprites both get tinted uniformly.
    # Cells in the player's current sight cone render crisp (no overlay);
    # previously-observed cells get a subtle memory tint; the rest is fog.
    fog_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
    fog_surf.fill(config.FOG_COLOR)
    seen_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
    seen_surf.fill(config.SEEN_TINT)

    sr = player.sight_range
    spotlight = {
        (player.x + dx, player.y + dy)
        for dx in range(-sr, sr + 1)
        for dy in range(-sr, sr + 1)
    }

    for gx in range(config.GRID_SIZE):
        for gy in range(config.GRID_SIZE):
            if (gx, gy) in spotlight:
                continue
            px = ox + gx * config.CELL_PX
            py = oy + gy * config.CELL_PX
            if (gx, gy) in player.observed_cells:
                screen.blit(seen_surf, (px, py))
            elif (config.NPC_OBSERVED_CELLS_VISIBLE or not config.PLAY_MODE) and (gx, gy) in brain.state.observed_cells:
                screen.blit(seen_surf, (px, py))
            else:
                screen.blit(fog_surf, (px, py))

    # NPC sight range highlight (debug only)
    if config.NPC_OBSERVED_CELLS_VISIBLE or not config.PLAY_MODE:
        sr = npc.sight_range
        sight_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
        sight_surf.fill((255, 200, 50, 25))
        for dx in range(-sr, sr + 1):
            for dy in range(-sr, sr + 1):
                gx, gy = npc.x + dx, npc.y + dy
                if 0 <= gx < config.GRID_SIZE and 0 <= gy < config.GRID_SIZE:
                    screen.blit(sight_surf, (ox + gx * config.CELL_PX, oy + gy * config.CELL_PX))


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
