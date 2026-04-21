"""
Replay viewer for saved game logs.

Loads a ``game.jsonl`` produced by ``GameLogger`` and reconstructs the
run frame by frame. Renders a full-visibility view of the world so the
reviewer can see everything — ground truth target, all shapes, both
sight cones, both explored regions — independent of the fog of war the
player experienced live.

Usage:
    python replay.py logs/runs/20260416T171539_seed42_play
    python replay.py logs/runs/20260416T171539_seed42_play/game.jsonl

Controls:
    LEFT / RIGHT                 step one frame back/forward
    SHIFT + LEFT / RIGHT         jump 10 frames
    HOME / END                   first / last frame
    SPACE                        play / pause autoplay
    UP / DOWN                    increase / decrease playback speed
    N                            jump to next interaction
    B                            jump to previous interaction
    1 / 2 / 3                    toggle NPC shading / player shading / sight cones
    I                            toggle interaction detail panel
    ESC                          quit

Design notes
────────────
* We build a ``Frame`` for every ``tick`` event plus a dedicated frame for
  every ``interaction_summary`` event. Interaction frames reuse the state
  of the most recent tick but attach the question/response payload so
  scrubbing naturally stops on them.
* ``npc_observed_cells`` / ``player_observed_cells`` / ``npc_shapes_seen``
  are folded cumulatively across tick deltas. Initial (pre-first-tick)
  observations are reconstructed from ``run_start`` geometry so frame 0
  already shows the starting sight cones.
* Shape positions and the natural-aliasing dicts come straight from
  ``run_start`` so replay never needs to import game code or re-run the
  world generator.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pygame


# ── Color palette (kept in sync with main.py) ─────────────────────────────────

SHAPE_COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (220, 60, 60),
    "blue":   (60, 120, 230),
    "green":  (60, 200, 90),
    "yellow": (230, 210, 50),
    "purple": (120, 80, 200),
}

BG_COLOR = (30, 30, 35)
GRID_LINE_COLOR = (50, 50, 55)
TEXT_COLOR = (210, 210, 210)
MUTED_TEXT = (140, 140, 150)
HIGHLIGHT_COLOR = (100, 200, 255)
NPC_COLOR = (255, 200, 50)
PLAYER_COLOR = (255, 255, 255)
TARGET_MARKER_COLOR = (255, 80, 180)
INTERACTION_MARKER_COLOR = (255, 180, 50)

NPC_EXPLORED_TINT = (255, 200, 50, 35)
PLAYER_EXPLORED_TINT = (120, 170, 255, 55)
NPC_SIGHT_TINT = (255, 200, 50, 70)
PLAYER_SIGHT_TINT = (200, 220, 255, 100)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Shape:
    label: str
    natural_name: str
    color: str
    shape_type: str
    x: int
    y: int
    is_target: bool


@dataclass
class RunMeta:
    """Static data extracted from the ``run_start`` event."""

    run_id: str
    tag: str
    seed: Optional[int]
    grid_size: int
    npc_start: tuple[int, int]
    npc_sight_range: int
    npc_brain_type: str
    npc_goal_label: Optional[str]
    player_start: tuple[int, int]
    player_sight_range: int
    target_label: str
    target_natural_name: str
    target_position: Optional[tuple[int, int]]
    shapes: list[Shape]
    config_snapshot: dict[str, Any]
    natural_aliases: dict[str, Any]
    extra_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    """One interaction_summary payload, indexed for sidebar rendering."""

    interaction_id: int
    tick: int
    frame_index: int
    question: str
    response_raw: str
    response_final: str
    response_mode: Optional[str]
    target_label: str
    target_natural_name: str
    target_position: Optional[tuple[int, int]]
    target_was_observed: bool
    npc_coverage: float
    grounding_violation: bool
    grounding_violations: list[tuple[int, int]]
    tool_calls: list[dict[str, Any]]
    token_usage_total: dict[str, int]
    llm_error: Optional[str]
    npc_context_lines: list[str]
    npc_shape_locations: dict[str, list[tuple[int, int]]]


@dataclass
class Frame:
    """One rendered state snapshot."""

    index: int
    tick: int
    npc_pos: tuple[int, int]
    player_pos: tuple[int, int]
    npc_observed_cells: frozenset[tuple[int, int]]
    player_observed_cells: frozenset[tuple[int, int]]
    npc_shapes_seen: frozenset[tuple[str, int, int]]
    npc_coverage: float
    npc_steps: int
    event_msg: Optional[str] = None
    interaction: Optional[Interaction] = None


# ── Log loading ───────────────────────────────────────────────────────────────


def _sight_cone(cx: int, cy: int, r: int, grid_size: int) -> set[tuple[int, int]]:
    return {
        (x, y)
        for x in range(max(0, cx - r), min(grid_size, cx + r + 1))
        for y in range(max(0, cy - r), min(grid_size, cy + r + 1))
    }


def _as_tuple(xy: Any) -> tuple[int, int]:
    return (int(xy[0]), int(xy[1]))


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSONL line {line_num}: {exc}", file=sys.stderr)
    return events


def _parse_run_start(event: dict[str, Any]) -> RunMeta:
    world = event.get("world", {})
    npc = event.get("npc", {})
    player = event.get("player", {})
    shapes = [
        Shape(
            label=s["label"],
            natural_name=s.get("natural_name", s["label"]),
            color=s["color"],
            shape_type=s["shape_type"],
            x=int(s["position"][0]),
            y=int(s["position"][1]),
            is_target=bool(s.get("is_target", False)),
        )
        for s in world.get("shapes", [])
    ]
    target_position = world.get("target_position")
    return RunMeta(
        run_id=event.get("run_id", "unknown"),
        tag=event.get("tag", ""),
        seed=event.get("seed"),
        grid_size=int(world.get("size", 15)),
        npc_start=_as_tuple(npc.get("start", [0, 0])),
        npc_sight_range=int(npc.get("sight_range", 1)),
        npc_brain_type=npc.get("brain_type", ""),
        npc_goal_label=npc.get("goal_label"),
        player_start=_as_tuple(player.get("start", [0, 0])),
        player_sight_range=int(player.get("sight_range", 2)),
        target_label=world.get("target_label", ""),
        target_natural_name=world.get("target_natural_name", ""),
        target_position=_as_tuple(target_position) if target_position else None,
        shapes=shapes,
        config_snapshot=event.get("config", {}),
        natural_aliases=event.get("natural_aliases", {}),
        extra_meta=event.get("extra_meta", {}),
    )


def _parse_interaction(event: dict[str, Any], frame_index: int) -> Interaction:
    resp = event.get("response", {}) or {}
    npc_knowledge = event.get("npc_knowledge", {}) or {}
    shape_locs: dict[str, list[tuple[int, int]]] = {
        label: [_as_tuple(c) for c in coords]
        for label, coords in (npc_knowledge.get("shape_locations", {}) or {}).items()
    }
    tp = event.get("target_position")
    return Interaction(
        interaction_id=int(event.get("interaction_id", -1)),
        tick=int(event.get("tick", -1)),
        frame_index=frame_index,
        question=event.get("question", ""),
        response_raw=resp.get("raw", ""),
        response_final=resp.get("final", ""),
        response_mode=resp.get("mode"),
        target_label=event.get("target_label", ""),
        target_natural_name=event.get("target_natural_name", ""),
        target_position=_as_tuple(tp) if tp else None,
        target_was_observed=bool(event.get("target_was_observed", False)),
        npc_coverage=float(event.get("npc_coverage", 0.0)),
        grounding_violation=bool(resp.get("grounding_violation", False)),
        grounding_violations=[_as_tuple(c) for c in (resp.get("grounding_violations") or [])],
        tool_calls=list(resp.get("tool_calls") or []),
        token_usage_total=dict(resp.get("token_usage_total") or {}),
        llm_error=resp.get("llm_error"),
        npc_context_lines=list(npc_knowledge.get("context_lines") or []),
        npc_shape_locations=shape_locs,
    )


def build_frames(events: list[dict[str, Any]]) -> tuple[RunMeta, list[Frame], list[Interaction]]:
    """
    Fold tick deltas into cumulative per-frame state.

    Raises ValueError if the first event is not ``run_start``.
    """
    if not events or events[0].get("event") != "run_start":
        raise ValueError("log must begin with a run_start event")

    meta = _parse_run_start(events[0])

    # Seed state with starting positions and initial sight cones. The first
    # ``tick`` event generally already contains the NPC's opening observation
    # batch, but we seed the player's sight cone here so frame 0 shows it.
    npc_pos = meta.npc_start
    player_pos = meta.player_start
    npc_observed: set[tuple[int, int]] = set()
    player_observed: set[tuple[int, int]] = _sight_cone(
        player_pos[0], player_pos[1], meta.player_sight_range, meta.grid_size
    )
    npc_shapes: set[tuple[str, int, int]] = set()
    npc_coverage = 0.0
    npc_steps = 0

    frames: list[Frame] = []
    interactions: list[Interaction] = []

    # Synthetic frame 0 — state at game start, before any tick fired.
    frames.append(
        Frame(
            index=0,
            tick=0,
            npc_pos=npc_pos,
            player_pos=player_pos,
            npc_observed_cells=frozenset(npc_observed),
            player_observed_cells=frozenset(player_observed),
            npc_shapes_seen=frozenset(npc_shapes),
            npc_coverage=0.0,
            npc_steps=0,
        )
    )

    for ev in events[1:]:
        etype = ev.get("event")
        if etype == "tick":
            tick_num = int(ev.get("tick", len(frames)))
            if (pos := ev.get("npc_pos")) is not None:
                npc_pos = _as_tuple(pos)
            if (pos := ev.get("player_pos")) is not None:
                player_pos = _as_tuple(pos)
                player_observed |= _sight_cone(
                    player_pos[0], player_pos[1], meta.player_sight_range, meta.grid_size
                )
            for cell in ev.get("npc_new_observed_cells", []) or []:
                npc_observed.add(_as_tuple(cell))
            for cell in ev.get("player_new_observed_cells", []) or []:
                player_observed.add(_as_tuple(cell))
            for sh in ev.get("npc_new_observed_shapes", []) or []:
                pos = sh.get("position") or (0, 0)
                npc_shapes.add((sh.get("label", ""), int(pos[0]), int(pos[1])))
            if (cov := ev.get("npc_coverage")) is not None:
                npc_coverage = float(cov)
            if (steps := ev.get("npc_steps")) is not None:
                npc_steps = int(steps)
            event_msg = ev.get("event_msg")

            frames.append(
                Frame(
                    index=len(frames),
                    tick=tick_num,
                    npc_pos=npc_pos,
                    player_pos=player_pos,
                    npc_observed_cells=frozenset(npc_observed),
                    player_observed_cells=frozenset(player_observed),
                    npc_shapes_seen=frozenset(npc_shapes),
                    npc_coverage=npc_coverage,
                    npc_steps=npc_steps,
                    event_msg=event_msg,
                )
            )

        elif etype == "interaction_summary":
            # Emit a dedicated frame so the timeline stops on the interaction.
            interaction = _parse_interaction(ev, frame_index=len(frames))
            interactions.append(interaction)
            frames.append(
                Frame(
                    index=len(frames),
                    tick=int(ev.get("tick", frames[-1].tick)),
                    npc_pos=npc_pos,
                    player_pos=player_pos,
                    npc_observed_cells=frozenset(npc_observed),
                    player_observed_cells=frozenset(player_observed),
                    npc_shapes_seen=frozenset(npc_shapes),
                    npc_coverage=npc_coverage,
                    npc_steps=npc_steps,
                    interaction=interaction,
                )
            )
        # All other event types (model_request, tool_call, run_end, ...) are
        # captured in the interaction_summary payload already; they do not
        # drive new frames.

    return meta, frames, interactions


# ── Rendering ─────────────────────────────────────────────────────────────────


LAYOUT = {
    "cell_px": 40,
    "top_bar_h": 50,
    "timeline_h": 42,
    "sidebar_w": 460,
    "fps": 30,
    "font_size": 14,
}


@dataclass
class ViewState:
    frame_index: int = 0
    playing: bool = False
    speed_ticks_per_sec: float = 4.0
    last_advance_ms: int = 0
    show_npc_explored: bool = True
    show_player_explored: bool = True
    show_sight_cones: bool = True
    show_interaction_panel: bool = True


def _grid_origin() -> tuple[int, int]:
    return (0, LAYOUT["top_bar_h"] + LAYOUT["timeline_h"])


def _window_size(meta: RunMeta) -> tuple[int, int]:
    grid_px = meta.grid_size * LAYOUT["cell_px"]
    w = grid_px + LAYOUT["sidebar_w"]
    h = LAYOUT["top_bar_h"] + LAYOUT["timeline_h"] + grid_px
    return w, h


def _draw_top_bar(screen, fonts, meta: RunMeta, view: ViewState, frames: list[Frame]) -> None:
    bar = pygame.Rect(0, 0, screen.get_width(), LAYOUT["top_bar_h"])
    pygame.draw.rect(screen, (40, 40, 50), bar)

    frame = frames[view.frame_index]
    playing = "▶" if view.playing else "⏸"
    title = (
        f"{meta.run_id}    {playing} {view.speed_ticks_per_sec:.1f}x    "
        f"frame {view.frame_index}/{len(frames) - 1}    tick {frame.tick}    "
        f"NPC steps {frame.npc_steps}    coverage {frame.npc_coverage:.0%}"
    )
    surf = fonts["big"].render(title, True, TEXT_COLOR)
    screen.blit(surf, (12, 15))


def _draw_timeline(screen, meta: RunMeta, view: ViewState,
                   frames: list[Frame], interactions: list[Interaction]) -> None:
    y = LAYOUT["top_bar_h"]
    h = LAYOUT["timeline_h"]
    w = screen.get_width()
    pygame.draw.rect(screen, (25, 25, 35), pygame.Rect(0, y, w, h))
    track_y = y + h // 2
    pygame.draw.line(screen, (80, 80, 95), (20, track_y), (w - 20, track_y), 2)

    def _x_for_index(i: int) -> int:
        if len(frames) <= 1:
            return 20
        return 20 + int((w - 40) * i / (len(frames) - 1))

    for ix in interactions:
        x = _x_for_index(ix.frame_index)
        pygame.draw.polygon(
            screen, INTERACTION_MARKER_COLOR,
            [(x, track_y - 8), (x + 6, track_y), (x, track_y + 8), (x - 6, track_y)],
        )

    x_cur = _x_for_index(view.frame_index)
    pygame.draw.circle(screen, HIGHLIGHT_COLOR, (x_cur, track_y), 7)
    pygame.draw.circle(screen, (20, 20, 30), (x_cur, track_y), 7, 2)


def _draw_grid(screen, fonts, meta: RunMeta, view: ViewState, frame: Frame) -> None:
    cell = LAYOUT["cell_px"]
    ox, oy = _grid_origin()
    grid_px = meta.grid_size * cell

    # Shaded cumulative-observed regions (optional).
    tint = pygame.Surface((cell, cell), pygame.SRCALPHA)
    if view.show_npc_explored:
        tint.fill(NPC_EXPLORED_TINT)
        for (gx, gy) in frame.npc_observed_cells:
            screen.blit(tint, (ox + gx * cell, oy + gy * cell))
    if view.show_player_explored:
        tint.fill(PLAYER_EXPLORED_TINT)
        for (gx, gy) in frame.player_observed_cells:
            screen.blit(tint, (ox + gx * cell, oy + gy * cell))

    # Grid lines.
    for i in range(meta.grid_size + 1):
        px = ox + i * cell
        pygame.draw.line(screen, GRID_LINE_COLOR, (px, oy), (px, oy + grid_px))
        py = oy + i * cell
        pygame.draw.line(screen, GRID_LINE_COLOR, (ox, py), (ox + grid_px, py))

    # Current sight cones.
    if view.show_sight_cones:
        cone = pygame.Surface((cell, cell), pygame.SRCALPHA)
        cone.fill(NPC_SIGHT_TINT)
        for (gx, gy) in _sight_cone(
            frame.npc_pos[0], frame.npc_pos[1], meta.npc_sight_range, meta.grid_size
        ):
            screen.blit(cone, (ox + gx * cell, oy + gy * cell))
        cone.fill(PLAYER_SIGHT_TINT)
        for (gx, gy) in _sight_cone(
            frame.player_pos[0], frame.player_pos[1], meta.player_sight_range, meta.grid_size
        ):
            screen.blit(cone, (ox + gx * cell, oy + gy * cell))

    # All shapes (full visibility). NPC-observed shapes get a bright outline;
    # unobserved shapes render dimmer so reviewers can tell at a glance what
    # the NPC could actually talk about.
    observed_shape_keys = {(lbl, x, y) for (lbl, x, y) in frame.npc_shapes_seen}
    for s in meta.shapes:
        cx = ox + s.x * cell + cell // 2
        cy = oy + s.y * cell + cell // 2
        r = cell // 3
        base = SHAPE_COLORS.get(s.color, (200, 200, 200))
        observed = (s.label, s.x, s.y) in observed_shape_keys
        color = base if observed else tuple(int(c * 0.55) for c in base)
        outline = (255, 255, 255) if observed else (120, 120, 120)

        if s.shape_type == "circle":
            pygame.draw.circle(screen, color, (cx, cy), r)
            pygame.draw.circle(screen, outline, (cx, cy), r, 2)
        elif s.shape_type == "triangle":
            pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
            pygame.draw.polygon(screen, color, pts)
            pygame.draw.polygon(screen, outline, pts, 2)
        elif s.shape_type == "square":
            rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, outline, rect, 2)

        if s.is_target:
            # Bright pink ring so the ground-truth target is unmistakable.
            pygame.draw.circle(screen, TARGET_MARKER_COLOR, (cx, cy), r + 6, 2)

    # NPC (diamond) and player (square) on top.
    npx = ox + frame.npc_pos[0] * cell + cell // 2
    npy = oy + frame.npc_pos[1] * cell + cell // 2
    d = cell // 3
    pts = [(npx, npy - d), (npx + d, npy), (npx, npy + d), (npx - d, npy)]
    pygame.draw.polygon(screen, NPC_COLOR, pts)
    pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
    screen.blit(fonts["sm"].render("NPC", True, NPC_COLOR), (npx - 12, npy - d - 14))

    ppx = ox + frame.player_pos[0] * cell + cell // 2
    ppy = oy + frame.player_pos[1] * cell + cell // 2
    pr = cell // 3
    prect = pygame.Rect(ppx - pr, ppy - pr, pr * 2, pr * 2)
    pygame.draw.rect(screen, PLAYER_COLOR, prect)
    pygame.draw.rect(screen, (100, 100, 100), prect, 2)
    pygame.draw.line(screen, (0, 0, 0), (ppx - pr // 2, ppy), (ppx + pr // 2, ppy), 2)
    pygame.draw.line(screen, (0, 0, 0), (ppx, ppy - pr // 2), (ppx, ppy + pr // 2), 2)
    screen.blit(fonts["sm"].render("YOU", True, PLAYER_COLOR), (ppx - 12, ppy - pr - 14))


def _wrap_text(text: str, font, max_px: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_sidebar(screen, fonts, meta: RunMeta, view: ViewState,
                  frames: list[Frame], interactions: list[Interaction]) -> None:
    cell = LAYOUT["cell_px"]
    sidebar_x = meta.grid_size * cell + 14
    sidebar_w = LAYOUT["sidebar_w"] - 28
    y = LAYOUT["top_bar_h"] + LAYOUT["timeline_h"] + 10
    frame = frames[view.frame_index]

    # Ground-truth block.
    header = fonts["title"].render("GROUND TRUTH", True, HIGHLIGHT_COLOR)
    screen.blit(header, (sidebar_x, y))
    y += 22

    target_txt = (
        f"target: {meta.target_natural_name}  "
        f"[{meta.target_label} @ {meta.target_position}]"
    )
    screen.blit(fonts["body"].render(target_txt, True, TEXT_COLOR), (sidebar_x, y))
    y += 18
    if meta.npc_goal_label:
        goal_txt = f"NPC goal: {meta.npc_goal_label} (brain={meta.npc_brain_type})"
    else:
        goal_txt = f"NPC brain: {meta.npc_brain_type} (no goal)"
    screen.blit(fonts["body"].render(goal_txt, True, MUTED_TEXT), (sidebar_x, y))
    y += 18
    cfg = meta.config_snapshot
    mode_txt = (
        f"mode: knowledge={cfg.get('NPC_KNOWLEDGE_MODE')}  "
        f"response={cfg.get('NPC_RESPONSE_MODE')}  "
        f"competing={cfg.get('NPC_COMPETING')}"
    )
    screen.blit(fonts["body"].render(mode_txt, True, MUTED_TEXT), (sidebar_x, y))
    y += 22

    # NPC knowledge at this frame.
    screen.blit(
        fonts["title"].render("NPC OBSERVATIONS (cumulative)", True, HIGHLIGHT_COLOR),
        (sidebar_x, y),
    )
    y += 22
    npc_facts = [
        f"coverage: {frame.npc_coverage:.1%}",
        f"observed cells: {len(frame.npc_observed_cells)}",
        f"shapes seen: {len(frame.npc_shapes_seen)}",
        f"NPC @ {frame.npc_pos}   player @ {frame.player_pos}",
    ]
    for line in npc_facts:
        screen.blit(fonts["body"].render(line, True, TEXT_COLOR), (sidebar_x, y))
        y += 18
    if frame.event_msg:
        for wrap in _wrap_text(f"> {frame.event_msg}", fonts["body"], sidebar_w):
            screen.blit(fonts["body"].render(wrap, True, (180, 220, 180)), (sidebar_x, y))
            y += 17
    y += 8

    # Interaction detail panel.
    if view.show_interaction_panel:
        screen.blit(
            fonts["title"].render(
                f"INTERACTIONS ({len(interactions)} total)",
                True, HIGHLIGHT_COLOR),
            (sidebar_x, y),
        )
        y += 22

        if not interactions:
            screen.blit(
                fonts["body"].render("no interactions in this run", True, MUTED_TEXT),
                (sidebar_x, y),
            )
            return

        # Show current-or-most-recent interaction in full, plus a compact list
        # of others.
        current = frame.interaction
        if current is None:
            prior = [ix for ix in interactions if ix.frame_index <= view.frame_index]
            current = prior[-1] if prior else None

        if current is not None:
            y = _draw_interaction_detail(screen, fonts, current, meta, sidebar_x, sidebar_w, y)
            y += 6

        screen.blit(
            fonts["body"].render("all interactions:", True, MUTED_TEXT),
            (sidebar_x, y),
        )
        y += 18
        for ix in interactions:
            marker = "→" if current is not None and ix.interaction_id == current.interaction_id else " "
            line = (
                f"{marker} #{ix.interaction_id}  tick {ix.tick}  "
                f"{'OBS' if ix.target_was_observed else 'BLIND':<5}  "
                f"{(ix.response_final or ix.response_raw or '')[:40]}"
            )
            color = TEXT_COLOR if marker == "→" else MUTED_TEXT
            screen.blit(fonts["body"].render(line, True, color), (sidebar_x, y))
            y += 16
            if y > screen.get_height() - 20:
                screen.blit(
                    fonts["body"].render("...", True, MUTED_TEXT), (sidebar_x, y)
                )
                return


def _draw_interaction_detail(screen, fonts, ix: Interaction, meta: RunMeta,
                             x: int, w: int, y: int) -> int:
    """Render the selected interaction in full. Returns the y below it."""
    heading = (
        f"#{ix.interaction_id} @ tick {ix.tick}   mode={ix.response_mode}"
    )
    screen.blit(fonts["body_bold"].render(heading, True, HIGHLIGHT_COLOR), (x, y))
    y += 19
    tgt = (
        f"target: {ix.target_natural_name} ({ix.target_label}) @ "
        f"{ix.target_position}   observed={ix.target_was_observed}"
    )
    for wrap in _wrap_text(tgt, fonts["body"], w):
        screen.blit(fonts["body"].render(wrap, True, (220, 200, 255)), (x, y))
        y += 16

    screen.blit(fonts["body_bold"].render("Q:", True, PLAYER_COLOR), (x, y))
    for wrap in _wrap_text(ix.question, fonts["body"], w - 24):
        screen.blit(fonts["body"].render(wrap, True, TEXT_COLOR), (x + 24, y))
        y += 16
    y += 2
    screen.blit(fonts["body_bold"].render("A:", True, NPC_COLOR), (x, y))
    for wrap in _wrap_text(ix.response_final or ix.response_raw or "(no response)",
                           fonts["body"], w - 24):
        screen.blit(fonts["body"].render(wrap, True, (180, 220, 180)), (x + 24, y))
        y += 16
    if ix.response_raw and ix.response_final and ix.response_raw != ix.response_final:
        # Grounding guard altered the response — show the original for audit.
        for wrap in _wrap_text(f"(raw: {ix.response_raw})", fonts["body"], w):
            screen.blit(fonts["body"].render(wrap, True, (180, 150, 150)), (x, y))
            y += 15

    if ix.grounding_violation:
        gv = ", ".join(str(c) for c in ix.grounding_violations)
        screen.blit(
            fonts["body"].render(f"grounding violations: {gv}", True, (255, 140, 140)),
            (x, y),
        )
        y += 16
    if ix.llm_error:
        screen.blit(
            fonts["body"].render(f"llm_error: {ix.llm_error[:60]}", True, (255, 140, 140)),
            (x, y),
        )
        y += 16
    if ix.tool_calls:
        screen.blit(
            fonts["body"].render(f"tool calls: {len(ix.tool_calls)}", True, MUTED_TEXT),
            (x, y),
        )
        y += 16
    if ix.token_usage_total:
        tu = ix.token_usage_total
        screen.blit(
            fonts["body"].render(
                f"tokens: prompt={tu.get('prompt_token_count','?')}  "
                f"resp={tu.get('candidates_token_count','?')}  "
                f"total={tu.get('total_token_count','?')}",
                True, MUTED_TEXT),
            (x, y),
        )
        y += 16
    return y


# ── Event handling ────────────────────────────────────────────────────────────


def _next_interaction_index(interactions: list[Interaction], current: int,
                            direction: int) -> Optional[int]:
    if not interactions:
        return None
    if direction > 0:
        for ix in interactions:
            if ix.frame_index > current:
                return ix.frame_index
        return interactions[-1].frame_index
    for ix in reversed(interactions):
        if ix.frame_index < current:
            return ix.frame_index
    return interactions[0].frame_index


def _handle_key(event, view: ViewState, frames: list[Frame],
                interactions: list[Interaction]) -> bool:
    """Return False when the key requested a quit."""
    key = event.key
    mods = event.mod
    shift = bool(mods & pygame.KMOD_SHIFT)

    if key == pygame.K_ESCAPE:
        return False
    if key == pygame.K_SPACE:
        view.playing = not view.playing
        view.last_advance_ms = pygame.time.get_ticks()
    elif key == pygame.K_LEFT:
        step = 10 if shift else 1
        view.frame_index = max(0, view.frame_index - step)
        view.playing = False
    elif key == pygame.K_RIGHT:
        step = 10 if shift else 1
        view.frame_index = min(len(frames) - 1, view.frame_index + step)
        view.playing = False
    elif key == pygame.K_HOME:
        view.frame_index = 0
        view.playing = False
    elif key == pygame.K_END:
        view.frame_index = len(frames) - 1
        view.playing = False
    elif key == pygame.K_UP:
        view.speed_ticks_per_sec = min(60.0, view.speed_ticks_per_sec * 1.5)
    elif key == pygame.K_DOWN:
        view.speed_ticks_per_sec = max(0.25, view.speed_ticks_per_sec / 1.5)
    elif key == pygame.K_n:
        nxt = _next_interaction_index(interactions, view.frame_index, direction=1)
        if nxt is not None:
            view.frame_index = nxt
            view.playing = False
    elif key == pygame.K_b:
        prv = _next_interaction_index(interactions, view.frame_index, direction=-1)
        if prv is not None:
            view.frame_index = prv
            view.playing = False
    elif key == pygame.K_1:
        view.show_npc_explored = not view.show_npc_explored
    elif key == pygame.K_2:
        view.show_player_explored = not view.show_player_explored
    elif key == pygame.K_3:
        view.show_sight_cones = not view.show_sight_cones
    elif key == pygame.K_i:
        view.show_interaction_panel = not view.show_interaction_panel
    return True


# ── Entry point ───────────────────────────────────────────────────────────────


def _resolve_log_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        p = p / "game.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"game log not found: {p}")
    return p


def run(log_path: Path) -> None:
    events = _read_events(log_path)
    meta, frames, interactions = build_frames(events)

    pygame.init()
    pygame.display.set_caption(f"Replay — {meta.run_id}")
    w, h = _window_size(meta)
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    fonts = {
        "sm":         pygame.font.SysFont("consolas", LAYOUT["font_size"] - 2),
        "body":       pygame.font.SysFont("consolas", LAYOUT["font_size"]),
        "body_bold":  pygame.font.SysFont("consolas", LAYOUT["font_size"], bold=True),
        "title":      pygame.font.SysFont("consolas", LAYOUT["font_size"], bold=True),
        "big":        pygame.font.SysFont("consolas", 16, bold=True),
    }

    view = ViewState()
    running = True

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if not _handle_key(event, view, frames, interactions):
                    running = False

        if view.playing and view.frame_index < len(frames) - 1:
            interval_ms = max(16, int(1000 / max(view.speed_ticks_per_sec, 0.001)))
            if now - view.last_advance_ms >= interval_ms:
                view.frame_index += 1
                view.last_advance_ms = now
                if view.frame_index >= len(frames) - 1:
                    view.playing = False

        screen.fill(BG_COLOR)
        _draw_top_bar(screen, fonts, meta, view, frames)
        _draw_timeline(screen, meta, view, frames, interactions)
        _draw_grid(screen, fonts, meta, view, frames[view.frame_index])
        _draw_sidebar(screen, fonts, meta, view, frames, interactions)

        pygame.display.flip()
        clock.tick(LAYOUT["fps"])

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved game.jsonl run.")
    parser.add_argument(
        "log",
        help="Path to a game.jsonl file or the run directory containing it.",
    )
    args = parser.parse_args()
    run(_resolve_log_path(args.log))


if __name__ == "__main__":
    main()
