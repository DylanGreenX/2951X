"""
Per-run game state logger for replay and eval audit.

Writes a single ``game.jsonl`` per run containing every event type mixed
together (``run_start``, ``tick``, ``interaction_*``, ``model_*``, ``tool_*``,
``run_end``). Ticks are delta-encoded — a tick event is only written when
something changed since the previous snapshot — so replay reconstructs full
state by folding deltas into a running copy.

The logger also retargets ``config.NPC_LLM_LOG_PATH`` to the per-run
``game.jsonl`` so ``InteractionManager._log_llm_event`` appends directly
into the same file. This keeps the full pipeline in one place:

    run_start                           (config + ground truth)
    tick   tick   tick   ...            (delta positions / observations)
    interaction_pre                     (ground truth at ask time)
    interaction_start   model_request
    tool_call tool_call model_response
    interaction_final                   (from InteractionManager)
    interaction_summary                 (our paired ground-truth + response)
    tick   tick   ...
    run_end                             (final outcome + stats)
    → summary.json written alongside

The ``interaction_summary`` event is the key audit record for the question
of whether the LLM receives game-irrelevant information. It pairs (a) what
the NPC actually knew, (b) the literal target coordinate, (c) the rendered
system prompt (by referencing the adjacent ``interaction_start`` event),
and (d) both the raw and grounded responses. With ``run_start`` capturing
the full ``NATURAL_OBJECTS`` / ``NATURAL_LOCATIONS`` aliasing dictionaries,
an audit script can diff "what the LLM saw" vs "what the internal label /
coord actually was" offline.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import config

if TYPE_CHECKING:
    from entities import NPC, Player
    from interaction import InteractionManager
    from npc_brain import NPCBrain
    from world import GameWorld


DEFAULT_LOG_DIR = "logs/runs"


@dataclass
class _LastSnapshot:
    """Previous observed state, used to compute per-tick deltas."""
    npc_pos: tuple[int, int] | None = None
    player_pos: tuple[int, int] | None = None
    npc_observed_cells: set = field(default_factory=set)
    npc_observed_shape_keys: set = field(default_factory=set)
    player_observed_cells: set = field(default_factory=set)
    coverage: float | None = None
    tick_index: int = -1


class GameLogger:
    """
    Writes a JSONL event stream for one game run.

    Call order:
        logger = GameLogger.start(world, player, npc, brain, ...)   # writes run_start
        # every game frame / tick:
        logger.log_tick(world, player, npc, brain)                  # delta only
        # on interaction:
        logger.log_interaction_pre(world, brain, target_color, target_shape)
        # ... InteractionManager runs and auto-logs model_* / tool_* / interaction_final ...
        logger.log_interaction_summary(interaction_manager, brain, world, target_color, target_shape)
        # on quit / win / trial end:
        logger.end(outcome, extra_stats=...)
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        tag: str,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.tag = tag
        self.log_path = run_dir / "game.jsonl"
        self.summary_path = run_dir / "summary.json"
        self._last = _LastSnapshot()
        self._tick_counter = 0
        self._interaction_counter = 0
        self._run_started_at = datetime.now(timezone.utc)
        self._closed = False

        # Redirect InteractionManager's LLM event log into this run's file.
        # Stash the previous value so close() can restore it.
        self._prev_llm_log_path: Optional[str] = getattr(
            config, "NPC_LLM_LOG_PATH", None
        )
        config.NPC_LLM_LOG_PATH = str(self.log_path)

    @classmethod
    def start(
        cls,
        world: "GameWorld",
        player: "Player",
        npc: "NPC",
        brain: "NPCBrain",
        *,
        tag: str = "play",
        seed: int | None = None,
        log_dir: str | Path = DEFAULT_LOG_DIR,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> "GameLogger":
        """
        Create a new per-run directory and write the ``run_start`` event.

        tag identifies the run type (``play`` / ``eval`` / ``experiment``) and
        is embedded in the run_id so logs are self-describing on disk.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        seed_part = f"seed{seed}" if seed is not None else uuid.uuid4().hex[:8]
        run_id = f"{timestamp}_{seed_part}_{tag}"
        root = Path(log_dir) if log_dir != DEFAULT_LOG_DIR else Path(
            getattr(config, "GAME_LOG_DIR", DEFAULT_LOG_DIR)
        )
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = cls(run_dir=run_dir, run_id=run_id, tag=tag)
        logger._write_run_start(world, player, npc, brain, seed=seed, extra_meta=extra_meta)
        # Seed the snapshot so the first tick records only what's new beyond start.
        logger._last = _LastSnapshot(
            npc_pos=(npc.x, npc.y),
            player_pos=(player.x, player.y),
            npc_observed_cells=set(brain.state.observed_cells),
            npc_observed_shape_keys=logger._shape_key_set(brain),
            player_observed_cells=set(player.observed_cells),
            coverage=brain.state.coverage,
            tick_index=0,
        )
        return logger

    # ── Public logging API ────────────────────────────────────────────────────

    def log_tick(
        self,
        world: "GameWorld",
        player: "Player",
        npc: "NPC",
        brain: "NPCBrain",
        *,
        event_msg: str | None = None,
        interaction_active: bool = False,
    ) -> None:
        """
        Write a tick event IFF something changed since the last snapshot.

        event_msg is the brain.tick() return value (e.g. "Collected crimson
        flag at (3, 3)!") — always logged when present, regardless of delta.
        """
        if self._closed:
            return
        self._tick_counter += 1

        delta: dict[str, Any] = {}

        npc_pos = (npc.x, npc.y)
        if npc_pos != self._last.npc_pos:
            delta["npc_pos"] = list(npc_pos)

        player_pos = (player.x, player.y)
        if player_pos != self._last.player_pos:
            delta["player_pos"] = list(player_pos)

        new_npc_cells = brain.state.observed_cells - self._last.npc_observed_cells
        if new_npc_cells:
            delta["npc_new_observed_cells"] = sorted(list(c) for c in new_npc_cells)

        new_player_cells = player.observed_cells - self._last.player_observed_cells
        if new_player_cells:
            delta["player_new_observed_cells"] = sorted(list(c) for c in new_player_cells)

        current_shape_keys = self._shape_key_set(brain)
        new_shape_keys = current_shape_keys - self._last.npc_observed_shape_keys
        if new_shape_keys:
            delta["npc_new_observed_shapes"] = [
                {"label": label, "position": [x, y]}
                for (label, x, y) in sorted(new_shape_keys)
            ]

        coverage = round(brain.state.coverage, 4)
        if self._last.coverage is None or abs(coverage - self._last.coverage) > 1e-6:
            delta["npc_coverage"] = coverage

        if event_msg:
            delta["event_msg"] = event_msg
        if interaction_active:
            delta["interaction_active"] = True

        # Skip empty deltas. No state moved → no tick event.
        if not delta and not event_msg:
            return

        self._write_event(
            "tick",
            tick=self._tick_counter,
            npc_steps=npc.steps_taken,
            **delta,
        )

        self._last = _LastSnapshot(
            npc_pos=npc_pos,
            player_pos=player_pos,
            npc_observed_cells=set(brain.state.observed_cells),
            npc_observed_shape_keys=current_shape_keys,
            player_observed_cells=set(player.observed_cells),
            coverage=coverage,
            tick_index=self._tick_counter,
        )

    def log_interaction_pre(
        self,
        world: "GameWorld",
        player: "Player",
        npc: "NPC",
        brain: "NPCBrain",
        target_color: str,
        target_shape: str,
    ) -> int:
        """
        Write an ``interaction_pre`` event immediately before calling
        InteractionManager.start_interaction. Returns an interaction_id that
        pairs this event with the later ``interaction_summary``.

        This is separate from ``interaction_summary`` so that the subsequent
        ``model_request`` / ``tool_call`` events auto-appended by
        InteractionManager fall between them — keeping the narrative of one
        interaction contiguous in the log.
        """
        if self._closed:
            return -1
        self._interaction_counter += 1
        interaction_id = self._interaction_counter

        target_label = f"{target_color}_{target_shape}"
        target_position = self._find_shape_position(world, target_label)
        npc_knowledge = self._snapshot_npc_knowledge(brain)

        self._write_event(
            "interaction_pre",
            interaction_id=interaction_id,
            tick=self._tick_counter,
            target_label=target_label,
            target_natural_name=self._natural_object_name(target_label),
            target_position=list(target_position) if target_position else None,
            target_natural_location=(
                self._natural_location_name(*target_position) if target_position else None
            ),
            npc_pos=[npc.x, npc.y],
            player_pos=[player.x, player.y],
            npc_knowledge=npc_knowledge,
            target_was_observed=brain.state.seen_label(target_label),
            response_mode=getattr(config, "NPC_RESPONSE_MODE", None),
            knowledge_mode=getattr(config, "NPC_KNOWLEDGE_MODE", None),
            competing=getattr(config, "NPC_COMPETING", False),
        )
        return interaction_id

    def log_interaction_summary(
        self,
        interaction_id: int,
        interaction_manager: "InteractionManager",
        brain: "NPCBrain",
        world: "GameWorld",
        target_color: str,
        target_shape: str,
        question: str,
        response: str,
    ) -> None:
        """
        Write the post-interaction audit event. Pairs ground truth at ask
        time with the final response (raw + grounded) so eval scripts can
        inspect each interaction without cross-referencing llm events.
        """
        if self._closed:
            return
        target_label = f"{target_color}_{target_shape}"
        target_position = self._find_shape_position(world, target_label)

        self._write_event(
            "interaction_summary",
            interaction_id=interaction_id,
            tick=self._tick_counter,
            target_label=target_label,
            target_natural_name=self._natural_object_name(target_label),
            target_position=list(target_position) if target_position else None,
            target_was_observed=brain.state.seen_label(target_label),
            npc_coverage=round(brain.state.coverage, 4),
            npc_knowledge=self._snapshot_npc_knowledge(brain),
            question=question,
            response={
                "raw": interaction_manager.last_raw_response,
                "final": interaction_manager.last_response or response,
                "mode": getattr(config, "NPC_RESPONSE_MODE", None),
                "grounding_violation": interaction_manager.last_grounding_violation,
                "grounding_violations": [
                    list(c) for c in interaction_manager.last_grounding_violations
                ],
                "tool_calls": interaction_manager.last_tool_calls,
                "token_usage_total": interaction_manager.last_token_usage_total,
                "llm_error": interaction_manager.last_llm_error,
            },
        )

    def log_custom(self, event_type: str, **payload: Any) -> None:
        """Escape hatch for callers that want to record extra events."""
        if self._closed:
            return
        self._write_event(event_type, tick=self._tick_counter, **payload)

    def end(
        self,
        outcome: str,
        *,
        extra_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Write ``run_end`` and a compact ``summary.json`` alongside the JSONL.
        Restores ``config.NPC_LLM_LOG_PATH`` to its pre-run value.
        """
        if self._closed:
            return
        ended_at = datetime.now(timezone.utc)
        duration_s = (ended_at - self._run_started_at).total_seconds()
        summary = {
            "run_id": self.run_id,
            "tag": self.tag,
            "outcome": outcome,
            "tick_count": self._tick_counter,
            "interaction_count": self._interaction_counter,
            "duration_seconds": duration_s,
            "started_at": self._run_started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
        }
        if extra_stats:
            summary["stats"] = extra_stats

        self._write_event("run_end", **summary)
        with self.summary_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(summary, ensure_ascii=True, indent=2))

        if self._prev_llm_log_path is not None:
            config.NPC_LLM_LOG_PATH = self._prev_llm_log_path
        self._closed = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_run_start(
        self,
        world: "GameWorld",
        player: "Player",
        npc: "NPC",
        brain: "NPCBrain",
        *,
        seed: int | None,
        extra_meta: Optional[dict[str, Any]],
    ) -> None:
        """
        Captures everything replay and audit tools need that does not change
        during the run: config snapshot, world ground truth (all shapes with
        both internal labels and natural aliases), the full NATURAL_*
        dictionaries, and NPC/player starts.
        """
        target_label = f"{world.target_color}_{world.target_shape}"
        target_position = self._find_shape_position(world, target_label)

        shapes_ground_truth = [
            {
                "label": s.label,
                "natural_name": self._natural_object_name(s.label),
                "color": s.color,
                "shape_type": s.shape_type,
                "position": [s.x, s.y],
                "natural_location": self._natural_location_name(s.x, s.y),
                "is_target": s.label == target_label and (s.x, s.y) == target_position,
            }
            for s in world.shapes
        ]

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "tag": self.tag,
            "seed": seed,
            "config": {
                "GRID_SIZE": config.GRID_SIZE,
                "NUM_OBJECTS": getattr(config, "NUM_OBJECTS", None),
                "COLORS": list(config.COLORS),
                "SHAPES": list(config.SHAPES),
                "PLAYER_START": list(config.PLAYER_START),
                "NPC_START": list(config.NPC_START),
                "PLAYER_SIGHT_RANGE": config.PLAYER_SIGHT_RANGE,
                "NPC_SIGHT_RANGE": config.NPC_SIGHT_RANGE,
                "NPC_TICK_INTERVAL": config.NPC_TICK_INTERVAL,
                "NPC_EXPLORATION_TICKS": getattr(config, "NPC_EXPLORATION_TICKS", None),
                "NPC_GOAL": config.NPC_GOAL,
                "NPC_GOAL_DETERMINISTIC": config.NPC_GOAL_DETERMINISTIC,
                "NPC_GOAL_COLOR": config.NPC_GOAL_COLOR,
                "NPC_GOAL_SHAPE": config.NPC_GOAL_SHAPE,
                "NPC_COMPETING": config.NPC_COMPETING,
                "NPC_KNOWLEDGE_MODE": getattr(config, "NPC_KNOWLEDGE_MODE", None),
                "NPC_RESPONSE_MODE": getattr(config, "NPC_RESPONSE_MODE", None),
                "NPC_ENFORCE_GROUNDING": getattr(config, "NPC_ENFORCE_GROUNDING", None),
                "NPC_LLM_MAX_TOOL_TURNS": getattr(config, "NPC_LLM_MAX_TOOL_TURNS", None),
                "NPC_LLM_MAX_OUTPUT_TOKENS": getattr(config, "NPC_LLM_MAX_OUTPUT_TOKENS", None),
                "NPC_LLM_TEMPERATURE": getattr(config, "NPC_LLM_TEMPERATURE", None),
                "NPC_MEMORY_DECAY_TICKS": getattr(config, "NPC_MEMORY_DECAY_TICKS", None),
                "NPC_SELECTIVE_ATTENTION": getattr(config, "NPC_SELECTIVE_ATTENTION", None),
                "DETERMINISTIC_TARGET": config.DETERMINISTIC_TARGET,
                "PLAY_MODE": config.PLAY_MODE,
            },
            "world": {
                "size": world.size,
                "target_color": world.target_color,
                "target_shape": world.target_shape,
                "target_label": target_label,
                "target_natural_name": self._natural_object_name(target_label),
                "target_position": list(target_position) if target_position else None,
                "shapes": shapes_ground_truth,
            },
            "natural_aliases": {
                "objects": dict(config.NATURAL_OBJECTS),
                "colors": dict(config.NATURAL_COLORS),
                "shapes": dict(config.NATURAL_SHAPES),
                "locations": {
                    f"{x},{y}": name
                    for (x, y), name in config.NATURAL_LOCATIONS.items()
                },
            },
            "npc": {
                "start": [npc.x, npc.y],
                "sight_range": npc.sight_range,
                "brain_type": type(brain).__name__,
                "goal_label": getattr(npc, "goal_label", None),
                "goal_natural_name": (
                    self._natural_object_name(npc.goal_label)
                    if getattr(npc, "goal_label", None) else None
                ),
            },
            "player": {
                "start": [player.x, player.y],
                "sight_range": player.sight_range,
            },
        }
        if extra_meta:
            payload["extra_meta"] = extra_meta

        self._write_event("run_start", **payload)

    def _write_event(self, event_type: str, **payload: Any) -> None:
        """Append one JSONL event with a UTC timestamp header."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "run_id": self.run_id,
            **payload,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._json_safe(record), ensure_ascii=True) + "\n")

    # ── Ground-truth / knowledge snapshots ────────────────────────────────────

    @staticmethod
    def _shape_key_set(brain: "NPCBrain") -> set[tuple[str, int, int]]:
        """
        Flatten shape_locations into a stable set of (label, x, y) keys so
        per-tick deltas can cheaply report newly-observed shapes without
        re-serializing the whole dict.
        """
        keys: set[tuple[str, int, int]] = set()
        for label, positions in brain.state.shape_locations.items():
            for x, y in positions:
                keys.add((label, int(x), int(y)))
        return keys

    @staticmethod
    def _snapshot_npc_knowledge(brain: "NPCBrain") -> dict[str, Any]:
        """Compact JSON snapshot of what the NPC currently knows."""
        return {
            "coverage": round(brain.state.coverage, 4),
            "observed_cells_count": len(brain.state.observed_cells),
            "explored_regions": dict(brain.state.explored_regions),
            "shape_locations": {
                label: [[int(x), int(y)] for x, y in positions]
                for label, positions in brain.state.shape_locations.items()
            },
            "context_lines": brain.state.to_llm_context(),
        }

    @staticmethod
    def _find_shape_position(
        world: "GameWorld", target_label: str
    ) -> tuple[int, int] | None:
        for shape in world.shapes:
            if shape.label == target_label:
                return (shape.x, shape.y)
        return None

    @staticmethod
    def _natural_object_name(label: str) -> str:
        from rlang_engine import get_natural_object_name
        return get_natural_object_name(label)

    @staticmethod
    def _natural_location_name(x: int, y: int) -> str:
        from rlang_engine import get_natural_location_name
        return get_natural_location_name(x, y)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert arbitrary SDK/dataclass objects into JSON-serializable form."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return cls._json_safe(value.model_dump(mode="json", exclude_none=True))
        return repr(value)
