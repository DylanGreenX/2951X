"""
Toy demo: NPC with RLang-grounded knowledge.

The NPC wanders aimlessly to begin and
accumulates incidental observations about the world. The bottom
panel shows the RLang state serialized as natural language strings —
exactly what would be injected into an LLM prompt.

Controls:
  Arrow keys — move player
  TAB — cycle focused NPC
  R — reset world
  ESC / close — quit
"""
import sys
import random
import pygame

import config
from world import GameWorld
from entities import Player
from interaction import InteractionManager
from llm import SLMClient
from pygame_game_api import PygameGameAPI
from game_log import GameLogger
from npc_party import NPCActor, NPCParty


SHAPE_COLORS = {
    "red":    (220, 60, 60),
    "blue":   (60, 120, 230),
    "green":  (60, 200, 90),
    "yellow": (230, 210, 50),
    "purple": (120, 80, 200),
}

NPC_DRAW_COLORS = [
    config.NPC_COLOR,
    (255, 140, 90),
    (120, 220, 180),
    (180, 160, 255),
    (255, 230, 120),
]


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

    player_target_label = f"{target_color}_{target_shape}"
    npc_party = NPCParty.from_config(world, player_target_label)

    interaction_manager = InteractionManager(api=PygameGameAPI.from_game(world, player, npc_party.brain_map), slm_client=slm_client)

    # Start the run log. GameLogger retargets config.NPC_LLM_LOG_PATH so all
    # LLM events auto-flow into the per-run game.jsonl for auditing.
    logger = GameLogger.start(world, player, npc_party, tag="play", seed=seed)

    return world, player, npc_party, interaction_manager, logger


def _default_focused_npc_id(npc_party: NPCParty) -> str:
    return npc_party.actors[0].npc_id


def _load_map_assets(grid_px: int) -> dict:
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
    world, player, npc_party, interaction_manager, logger = _init_game(seed=42, slm_client=shared_slm_client)
    focused_npc_id = _default_focused_npc_id(npc_party)

    event_log: list[str] = []
    in_interaction = False
    interaction_question = ""
    interaction_response = ""

    last_npc_tick = pygame.time.get_ticks()
    running = True
    outcome = "quit"

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if in_interaction:
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        in_interaction = False
                    continue

                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                if event.key == pygame.K_TAB:
                    focused_npc_id = npc_party.next_actor_id(focused_npc_id)
                    continue
                if event.key == pygame.K_p:
                    npc_party.actor_by_id(focused_npc_id).brain.set_target_pos((player.x, player.y))
                    continue
                if event.key == pygame.K_r:
                    world, player, npc_party, interaction_manager, logger = _init_game(
                        prev_logger=logger, slm_client=shared_slm_client
                    )
                    focused_npc_id = _default_focused_npc_id(npc_party)
                    event_log.clear()
                    in_interaction = False
                    continue

                dx, dy = 0, 0
                if event.key == pygame.K_UP:
                    dy = -1
                elif event.key == pygame.K_DOWN:
                    dy = 1
                elif event.key == pygame.K_LEFT:
                    dx = -1
                elif event.key == pygame.K_RIGHT:
                    dx = 1
                if dx == 0 and dy == 0:
                    continue

                nx, ny = player.x + dx, player.y + dy
                if not world.in_bounds(nx, ny):
                    continue

                player.x, player.y = nx, ny
                world.update_player_vision(player)
                logger.log_tick(world, player, npc_party)

                interaction_actor = npc_party.actor_at(player.x, player.y)
                if interaction_actor is None:
                    continue

                focused_npc_id = interaction_actor.npc_id
                focused_actor = npc_party.actor_by_id(focused_npc_id)
                _render_loading_state(
                    screen,
                    font_big,
                    font_title,
                    font,
                    world,
                    player,
                    npc_party,
                    focused_actor,
                    event_log,
                    grid_px,
                    win_w,
                    assets,
                    message=f"{focused_actor.display_name} is thinking...",
                )

                interaction_id = logger.log_interaction_pre(
                    world, player, interaction_actor,
                    world.target_color, world.target_shape,
                )
                interaction_question, interaction_response = interaction_manager.start_interaction(
                    interaction_actor.brain,
                    world.target_color,
                    world.target_shape,
                )
                logger.log_interaction_summary(
                    interaction_id, interaction_manager, interaction_actor, world,
                    world.target_color, world.target_shape,
                    interaction_question, interaction_response,
                )
                in_interaction = True

        if not in_interaction and now - last_npc_tick >= config.NPC_TICK_INTERVAL:
            tick_result = npc_party.tick()
            for entry in tick_result.event_msgs:
                event_log.append(f"{entry['display_name']}: {entry['message']}")
            if len(event_log) > 8:
                event_log = event_log[-8:]
            logger.log_tick(
                world,
                player,
                npc_party,
                event_msgs=tick_result.event_msgs,
                knowledge_exchanges=tick_result.knowledge_exchanges,
            )
            last_npc_tick = now

        focused_actor = npc_party.actor_by_id(focused_npc_id)
        _render_scene(
            screen,
            font_big,
            font_title,
            font,
            world,
            player,
            npc_party,
            focused_actor,
            event_log,
            grid_px,
            win_w,
            assets,
        )
        if in_interaction:
            _draw_interaction_overlay(
                screen, font_big, font, interaction_question, interaction_response, win_w, win_h
            )

        pygame.display.flip()
        clock.tick(config.FPS)

    logger.end(
        outcome=outcome,
        extra_stats={
            "npc_steps_total": sum(actor.npc.steps_taken for actor in npc_party.actors),
            "npc_combined_coverage": round(npc_party.combined_coverage(), 4),
            "player_pos": [player.x, player.y],
            "npcs": [
                {
                    "npc_id": actor.npc_id,
                    "pos": [actor.npc.x, actor.npc.y],
                    "steps_taken": actor.npc.steps_taken,
                    "coverage": round(actor.brain.state.coverage, 4),
                }
                for actor in npc_party.actors
            ],
        },
    )
    pygame.quit()
    sys.exit()


def _render_scene(
    screen,
    font_big,
    font_title,
    font,
    world,
    player,
    npc_party: NPCParty,
    focused_actor: NPCActor,
    event_log: list[str],
    grid_px: int,
    win_w: int,
    assets: dict,
) -> None:
    screen.fill(config.BG_COLOR)
    _draw_top_bar(screen, font_big, npc_party, focused_actor)
    _draw_grid(screen, world, npc_party, focused_actor, player, font, assets)
    _draw_sidebar(
        screen, font_title, font, focused_actor, event_log, grid_px, len(npc_party.actors)
    )
    _draw_hud(screen, font_title, font, focused_actor, grid_px, win_w)


def _render_loading_state(
    screen,
    font_big,
    font_title,
    font,
    world,
    player,
    npc_party: NPCParty,
    focused_actor: NPCActor,
    event_log: list[str],
    grid_px: int,
    win_w: int,
    assets: dict,
    *,
    message: str,
) -> None:
    _render_scene(
        screen,
        font_big,
        font_title,
        font,
        world,
        player,
        npc_party,
        focused_actor,
        event_log,
        grid_px,
        win_w,
        assets,
    )
    _draw_loading_overlay(screen, font_big, font, message, screen.get_width(), screen.get_height())
    pygame.event.pump()
    pygame.display.flip()


def _grid_origin():
    return (0, config.TOP_BAR_H)


def _npc_draw_color(actor: NPCActor) -> tuple[int, int, int]:
    index = int(actor.npc_id.split("_")[-1])
    return NPC_DRAW_COLORS[index % len(NPC_DRAW_COLORS)]


def _draw_top_bar(screen, font, npc_party: NPCParty, focused_actor: NPCActor):
    bar = pygame.Rect(0, 0, screen.get_width(), config.TOP_BAR_H)
    pygame.draw.rect(screen, (40, 40, 50), bar)

    txt = (
        f"NPCs: {len(npc_party.actors)} ({npc_party.multiple_knowledge_mode})    "
        f"Focus: {focused_actor.display_name}    "
        f"Goal: {focused_actor.npc.goal_label or 'wandering'}    "
        f"Steps: {focused_actor.npc.steps_taken}    "
        f"Focus Explored: {focused_actor.brain.state.coverage:.0%}    "
        f"Combined Explored: {npc_party.combined_coverage():.0%}"
    )
    surf = font.render(txt, True, config.TEXT_COLOR)
    screen.blit(surf, (12, 15))


def _draw_grid(screen, world, npc_party: NPCParty, focused_actor: NPCActor, player, font, assets):
    ox, oy = _grid_origin()
    grid_px = config.GRID_SIZE * config.CELL_PX
    focused_cells = focused_actor.brain.state.observed_cells

    if assets.get("bg") is not None:
        screen.blit(assets["bg"], (ox, oy))

    for i in range(config.GRID_SIZE + 1):
        px = ox + i * config.CELL_PX
        pygame.draw.line(screen, config.GRID_LINE_COLOR, (px, oy), (px, oy + grid_px))
        py = oy + i * config.CELL_PX
        pygame.draw.line(screen, config.GRID_LINE_COLOR, (ox, py), (ox + grid_px, py))

    shape_sprites = assets.get("shapes", {})
    for shape in world.shapes:
        if shape.collected:
            continue
        visible = (
            (config.PLAY_MODE and (shape.x, shape.y) in player.observed_cells)
            or not config.PLAY_MODE
            or (config.NPC_OBSERVED_CELLS_VISIBLE and (shape.x, shape.y) in focused_cells)
        )
        if not visible:
            continue

        label = f"{shape.color}_{shape.shape_type}"
        sprite = shape_sprites.get(label)
        if sprite is not None:
            ix = ox + shape.x * config.CELL_PX + (config.CELL_PX - sprite.get_width()) // 2
            iy = oy + shape.y * config.CELL_PX + (config.CELL_PX - sprite.get_height()) // 2
            screen.blit(sprite, (ix, iy))
            continue

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

    npc_img = assets.get("npc")
    for actor in npc_party.actors:
        npc = actor.npc
        npc_visible = (npc.x, npc.y) in player.observed_cells or not config.PLAY_MODE
        if not npc_visible:
            continue
        npc_color = _npc_draw_color(actor)
        label_text = f"N{int(actor.npc_id.split('_')[-1]) + 1}"
        if npc_img is not None:
            ix = ox + npc.x * config.CELL_PX + (config.CELL_PX - npc_img.get_width()) // 2
            iy = oy + npc.y * config.CELL_PX + (config.CELL_PX - npc_img.get_height()) // 2
            screen.blit(npc_img, (ix, iy))
        else:
            npx = ox + npc.x * config.CELL_PX + config.CELL_PX // 2
            npy = oy + npc.y * config.CELL_PX + config.CELL_PX // 2
            d = config.CELL_PX // 3
            pts = [(npx, npy - d), (npx + d, npy), (npx, npy + d), (npx - d, npy)]
            pygame.draw.polygon(screen, npc_color, pts)
            pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
        label = font.render(label_text, True, npc_color)
        lx = ox + npc.x * config.CELL_PX + (config.CELL_PX - label.get_width()) // 2
        ly = oy + npc.y * config.CELL_PX - 14
        screen.blit(label, (lx, ly))

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
            elif (config.NPC_OBSERVED_CELLS_VISIBLE or not config.PLAY_MODE) and (gx, gy) in focused_cells:
                screen.blit(seen_surf, (px, py))
            else:
                screen.blit(fog_surf, (px, py))

    if config.NPC_OBSERVED_CELLS_VISIBLE or not config.PLAY_MODE:
        npc = focused_actor.npc
        sr = npc.sight_range
        sight_surf = pygame.Surface((config.CELL_PX, config.CELL_PX), pygame.SRCALPHA)
        sight_surf.fill((*_npc_draw_color(focused_actor), 25))
        for dx in range(-sr, sr + 1):
            for dy in range(-sr, sr + 1):
                gx, gy = npc.x + dx, npc.y + dy
                if 0 <= gx < config.GRID_SIZE and 0 <= gy < config.GRID_SIZE:
                    screen.blit(sight_surf, (ox + gx * config.CELL_PX, oy + gy * config.CELL_PX))


def _draw_sidebar(screen, font_title, font, focused_actor: NPCActor, event_log, grid_px, npc_count: int):
    ox = config.GRID_SIZE * config.CELL_PX + 10
    oy = config.TOP_BAR_H + 10

    title = font_title.render("LEGEND", True, config.HIGHLIGHT_COLOR)
    screen.blit(title, (ox, oy))
    oy += 24

    legend_items = [
        ("NPCs (numbered diamonds)", config.NPC_COLOR),
        ("Player (white square)", config.PLAYER_COLOR),
        ("Blue circle", SHAPE_COLORS["blue"]),
        ("Red triangle", SHAPE_COLORS["red"]),
        ("Green square", SHAPE_COLORS["green"]),
        ("Yellow triangle", SHAPE_COLORS["yellow"]),
        ("Bright cells = focused NPC explored", (80, 80, 90)),
        ("Dark cells = unexplored (fog)", (40, 40, 45)),
    ]
    for text, color in legend_items:
        pygame.draw.rect(screen, color, (ox, oy + 2, 12, 12))
        surf = font.render(text, True, config.TEXT_COLOR)
        screen.blit(surf, (ox + 18, oy))
        oy += 20

    oy += 8
    focus_text = font.render(
        f"Focused: {focused_actor.display_name}", True, _npc_draw_color(focused_actor)
    )
    screen.blit(focus_text, (ox, oy))
    oy += 20
    if npc_count > 1:
        tab_text = font.render("TAB cycles focused NPC", True, config.TEXT_COLOR)
        screen.blit(tab_text, (ox, oy))
        oy += 20

    oy += 16
    title = font_title.render("NPC EVENT LOG", True, config.HIGHLIGHT_COLOR)
    screen.blit(title, (ox, oy))
    oy += 22

    for entry in event_log[-8:]:
        surf = font.render(f"> {entry}", True, (180, 220, 180))
        screen.blit(surf, (ox, oy))
        oy += 18


def _draw_hud(screen, font_title, font, focused_actor: NPCActor, grid_px, win_w):
    oy = config.TOP_BAR_H + grid_px + 4

    hud_rect = pygame.Rect(0, oy, win_w, config.HUD_H)
    pygame.draw.rect(screen, (25, 25, 35), hud_rect)
    pygame.draw.line(screen, config.HIGHLIGHT_COLOR, (0, oy), (win_w, oy), 2)

    oy += 6
    title = font_title.render(
        f"  LLM CONTEXT ({focused_actor.display_name})", True, config.HIGHLIGHT_COLOR
    )
    screen.blit(title, (4, oy))
    oy += 22

    for line in focused_actor.brain.state.to_llm_context():
        if "[PROPOSITION]" in line:
            color = (180, 180, 220)
        elif "[OBSERVATION]" in line:
            color = (180, 220, 180)
        elif "[FACTOR]" in line:
            color = (220, 200, 150)
        else:
            color = config.TEXT_COLOR
        display = line if len(line) < 90 else line[:87] + "..."
        surf = font.render(display, True, color)
        screen.blit(surf, (12, oy))
        oy += 17
        if oy > config.TOP_BAR_H + grid_px + config.HUD_H - 10:
            surf = font.render("  ... (more)", True, (120, 120, 120))
            screen.blit(surf, (12, oy))
            break


def _draw_loading_overlay(screen, font_big, font, message: str, win_w: int, win_h: int):
    box_w, box_h = 420, 120
    box_x = (win_w - box_w) // 2
    box_y = (win_h - box_h) // 2

    dim = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 140))
    screen.blit(dim, (0, 0))

    pygame.draw.rect(screen, (30, 30, 45), (box_x, box_y, box_w, box_h), border_radius=8)
    pygame.draw.rect(screen, config.HIGHLIGHT_COLOR, (box_x, box_y, box_w, box_h), 2, border_radius=8)

    label = font_big.render("PLEASE WAIT", True, config.HIGHLIGHT_COLOR)
    screen.blit(label, (box_x + 18, box_y + 18))
    msg = font.render(message, True, config.TEXT_COLOR)
    screen.blit(msg, (box_x + 18, box_y + 58))


def _draw_interaction_overlay(screen, font_big, font, question: str, response: str, win_w: int, win_h: int):
    box_w, box_h = 540, 180
    box_x = (win_w - box_w) // 2
    box_y = (win_h - box_h) // 2

    dim = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 150))
    screen.blit(dim, (0, 0))

    pygame.draw.rect(screen, (30, 30, 45), (box_x, box_y, box_w, box_h), border_radius=8)
    pygame.draw.rect(screen, config.HIGHLIGHT_COLOR, (box_x, box_y, box_w, box_h), 2, border_radius=8)

    pad = 18
    y = box_y + pad

    you_label = font_big.render("YOU:", True, config.PLAYER_COLOR)
    screen.blit(you_label, (box_x + pad, y))
    q_surf = font.render(question, True, config.TEXT_COLOR)
    screen.blit(q_surf, (box_x + pad + you_label.get_width() + 8, y + 3))
    y += you_label.get_height() + 14

    npc_label = font_big.render("NPC:", True, config.NPC_COLOR)
    screen.blit(npc_label, (box_x + pad, y))
    max_chars = (box_w - pad * 2 - npc_label.get_width() - 8) // 8
    words = response.split()
    lines: list[str] = []
    current = ""
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

    hint = font.render("[ ENTER / ESC ]  to close", True, (120, 120, 140))
    screen.blit(hint, (box_x + box_w - hint.get_width() - pad, box_y + box_h - hint.get_height() - 10))


if __name__ == "__main__":
    main()
