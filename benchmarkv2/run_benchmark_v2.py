from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import config
import metrics
from entities import Player
from game_log import GameLogger
from interaction import InteractionManager
from llm import SLMClient
from npc_party import NPCActor, NPCParty
from pygame_game_api import PygameGameAPI
from rlang_engine import get_natural_object_name
from world import GameWorld


OUTPUT_ROOT = Path("benchmarkv2/results")
DEFAULT_GRID_SIZES = [15, 40]
DEFAULT_PARTY_SIZES = [2, 4]
DEFAULT_RESPONSE_MODES = ["deterministic", "llm", "slm"]
DEFAULT_NUM_TRIALS = 50
BASE_SEED = 20260429
BENCHMARK_NPC_SIGHT_RANGE = 1
BENCHMARK_PLAYER_SIGHT_RANGE = 2
BENCHMARK_EXPLORATION_TICKS = 40
BENCHMARK_EXPLORATION_TICKS_BY_GRID_AND_NPCS = {
    (15, 1): 70,
    (15, 2): 40,
    (15, 4): 12,
    (40, 1): 1500,
    (40, 2): 450,
    (40, 4): 240,
}


@dataclass(frozen=True)
class BenchmarkV2Condition:
    name: str
    scope: str  # "single" | "party"
    npc_count: int
    knowledge_mode: str  # "perfect" | "embodied"
    response_mode: str  # "deterministic" | "llm" | "slm"
    grid_size: int
    competitor_mode: str = "none"  # "none" | "one" | "all"

    @property
    def competitor_count(self) -> int:
        if self.competitor_mode == "one":
            return 1
        if self.competitor_mode == "all":
            return self.npc_count
        return 0


@dataclass
class _SavedConfig:
    values: dict[str, Any]
    missing: set[str]


CONFIG_KEYS = {
    "GRID_SIZE",
    "NPC_COUNT",
    "NPC_START",
    "NPC_STARTS",
    "NPC_SIGHT_RANGE",
    "PLAYER_START",
    "PLAYER_SIGHT_RANGE",
    "NPC_EXPLORATION_TICKS",
    "NPC_GOAL",
    "NPC_GOAL_DETERMINISTIC",
    "NPC_GOAL_COLOR",
    "NPC_GOAL_SHAPE",
    "NPC_KNOWLEDGE_MODE",
    "NPC_RESPONSE_MODE",
    "NPC_COMPETING",
    "NPC_COMPETING_COUNT",
    "NPC_MULTIPLE_KNOWLEDGE_MODE",
    "NPC_NONCOMPETING_GOAL_MODE",
    "NPC_NPC_INTERACTION_ENABLED",
    "NPC_MEMORY_DECAY_TICKS",
    "NPC_SELECTIVE_ATTENTION",
    "NPC_SLM_REGION_GROUNDING",
    "NPC_USE_LLM_JUDGE",
    "RANDOM_SPAWN",
    "WORLD_RESERVED_POSITIONS",
}


@contextmanager
def temporary_config(**overrides: Any):
    keys = CONFIG_KEYS | set(overrides)
    missing = {key for key in keys if not hasattr(config, key)}
    saved = _SavedConfig(
        values={key: getattr(config, key) for key in keys if key not in missing},
        missing=missing,
    )
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        _refresh_region_phrases()
        yield
    finally:
        for key in overrides:
            if key in saved.missing and hasattr(config, key):
                delattr(config, key)
        for key, value in saved.values.items():
            setattr(config, key, value)
        _refresh_region_phrases()


def _refresh_region_phrases() -> None:
    # rlang_engine computes its extraction phrase table at import time from
    # config.GRID_SIZE. Benchmark v2 can vary grid size, so keep it synchronized.
    import rlang_engine

    rlang_engine._REGION_PHRASES = rlang_engine._collect_region_phrases(  # noqa: SLF001
        config.GRID_SIZE
    )


def corner_positions(world_size: int) -> list[tuple[int, int]]:
    margin = 1 if world_size > 2 else 0
    far = max(0, world_size - 1 - margin)
    return [(margin, margin), (far, margin), (margin, far), (far, far)]


def benchmark_reserved_positions(world_size: int) -> list[tuple[int, int]]:
    """Stable reserved cells so object placement stays paired across conditions."""
    margin = 1 if world_size > 2 else 0
    far = max(0, world_size - 1 - margin)
    mid = world_size // 2
    anchors = [
        *corner_positions(world_size),
        (mid, margin),
        (mid, far),
        (margin, mid),
        (far, mid),
        (mid, mid),
    ]
    return list(dict.fromkeys(anchors))


def exploration_ticks_for(grid_size: int, npc_count: int) -> int:
    return BENCHMARK_EXPLORATION_TICKS_BY_GRID_AND_NPCS.get(
        (grid_size, npc_count),
        BENCHMARK_EXPLORATION_TICKS,
    )


def spread_start_positions(
    *,
    rng: random.Random,
    npc_count: int,
    world_size: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Choose separated NPC/player starts from stable anchors, randomized by seed."""
    if npc_count < 1:
        raise ValueError("npc_count must be positive")

    corners = corner_positions(world_size)
    rng.shuffle(corners)
    npc_starts = corners[: min(npc_count, len(corners))]

    remaining = [pos for pos in corners if pos not in npc_starts]
    if remaining:
        player_start = rng.choice(remaining)
        return npc_starts, player_start

    mid = world_size // 2
    margin = 1 if world_size > 2 else 0
    far = max(0, world_size - 1 - margin)
    fallback = [(mid, margin), (mid, far), (margin, mid), (far, mid), (mid, mid)]
    fallback = [pos for pos in fallback if pos not in npc_starts]
    rng.shuffle(fallback)
    if not fallback:
        raise ValueError(f"Unable to place player with {npc_count} NPCs")
    return npc_starts, fallback[0]


class TargetAvailabilityTracker:
    def __init__(
        self,
        target_label: str,
        target_location: tuple[int, int] | None,
    ) -> None:
        self.target_label = target_label
        self.target_location = target_location
        self.actor_first_available: dict[str, int] = {}
        self.actor_first_direct: dict[str, int] = {}
        self.actor_first_learned: dict[str, int] = {}
        self.actor_first_source_kind: dict[str, str] = {}
        self.actor_first_source_npcs: dict[str, tuple[str, ...]] = {}

    def record(self, party: NPCParty, tick: int) -> None:
        if self.target_location is None:
            return
        for actor in party.actors:
            source = self._target_source(actor)
            if source is None:
                continue
            kind, source_npcs = source
            self.actor_first_available.setdefault(actor.npc_id, tick)
            self.actor_first_source_kind.setdefault(actor.npc_id, kind)
            self.actor_first_source_npcs.setdefault(actor.npc_id, source_npcs)
            if kind in {"direct", "perfect"}:
                self.actor_first_direct.setdefault(actor.npc_id, tick)
            if kind == "interaction":
                self.actor_first_learned.setdefault(actor.npc_id, tick)

    def _target_source(self, actor: NPCActor) -> tuple[str, tuple[str, ...]] | None:
        positions = actor.brain.state.shape_locations.get(self.target_label, [])
        if not positions:
            return None
        target_pos = self.target_location
        if target_pos not in positions:
            target_pos = positions[0]
        kind, source_npcs = actor.brain.state.observed_shape_sources.get(
            target_pos, ("direct", ())
        )
        return kind, tuple(source_npcs)

    def summary(self, party: NPCParty, speaker_id: str) -> dict[str, Any]:
        party_available_ticks = list(self.actor_first_available.values())
        party_direct_ticks = list(self.actor_first_direct.values())
        party_learned_ticks = list(self.actor_first_learned.values())
        speaker_source_npcs = self.actor_first_source_npcs.get(speaker_id)
        return {
            "speaker_target_available": speaker_id in self.actor_first_available,
            "speaker_target_available_tick": self.actor_first_available.get(speaker_id),
            "speaker_target_direct_tick": self.actor_first_direct.get(speaker_id),
            "speaker_target_learned_tick": self.actor_first_learned.get(speaker_id),
            "speaker_target_source_kind": self.actor_first_source_kind.get(speaker_id),
            "speaker_target_source_npcs": (
                list(speaker_source_npcs) if speaker_source_npcs is not None else None
            ),
            "party_target_available": bool(party_available_ticks),
            "party_target_available_tick": min(party_available_ticks) if party_available_ticks else None,
            "party_target_direct_tick": min(party_direct_ticks) if party_direct_ticks else None,
            "party_target_learned_tick": min(party_learned_ticks) if party_learned_ticks else None,
            "party_target_available_npcs": sorted(self.actor_first_available),
            "party_target_direct_npcs": sorted(self.actor_first_direct),
            "party_target_learned_npcs": sorted(self.actor_first_learned),
        }


def build_conditions(
    *,
    grid_sizes: Iterable[int] = DEFAULT_GRID_SIZES,
    response_modes: Iterable[str] = DEFAULT_RESPONSE_MODES,
    party_sizes: Iterable[int] = DEFAULT_PARTY_SIZES,
) -> list[BenchmarkV2Condition]:
    response_modes = list(response_modes)
    conditions: list[BenchmarkV2Condition] = []
    for grid_size in grid_sizes:
        for knowledge_mode in ("perfect", "embodied"):
            for response_mode in response_modes:
                conditions.append(
                    BenchmarkV2Condition(
                        name=f"v2 single {grid_size} {knowledge_mode} {response_mode}",
                        scope="single",
                        npc_count=1,
                        knowledge_mode=knowledge_mode,
                        response_mode=response_mode,
                        grid_size=grid_size,
                    )
                )

        for party_size in party_sizes:
            for response_mode in response_modes:
                conditions.append(
                    BenchmarkV2Condition(
                        name=f"v2 party{party_size} {grid_size} embodied {response_mode} none",
                        scope="party",
                        npc_count=party_size,
                        knowledge_mode="embodied",
                        response_mode=response_mode,
                        grid_size=grid_size,
                    )
                )

            for competitor_mode in ("one", "all"):
                for response_mode in response_modes:
                    if response_mode == "deterministic":
                        continue
                    conditions.append(
                        BenchmarkV2Condition(
                            name=(
                                f"v2 party{party_size} {grid_size} embodied "
                                f"{response_mode} competitor-{competitor_mode}"
                            ),
                            scope="party",
                            npc_count=party_size,
                            knowledge_mode="embodied",
                            response_mode=response_mode,
                            grid_size=grid_size,
                            competitor_mode=competitor_mode,
                        )
                    )
    return conditions


class BenchmarkV2Runner:
    def __init__(
        self,
        *,
        output_root: Path = OUTPUT_ROOT,
        use_judge: bool = False,
        base_seed: int = BASE_SEED,
    ) -> None:
        self.output_root = output_root
        self.use_judge = use_judge
        self.base_seed = base_seed
        self._slm_client: SLMClient | None = None

    def _get_slm_client(self) -> SLMClient:
        if self._slm_client is None:
            self._slm_client = SLMClient(
                model_id=getattr(config, "NPC_SLM_MODEL_ID", "HuggingFaceTB/SmolLM-135M"),
                device=getattr(config, "NPC_SLM_DEVICE", "auto"),
                dtype=getattr(config, "NPC_SLM_DTYPE", "auto"),
            )
            self._slm_client.preload()
        return self._slm_client

    def run(
        self,
        *,
        conditions: list[BenchmarkV2Condition],
        num_trials: int,
    ) -> Path:
        run_dir = self._create_output_dir()
        csv_path = run_dir / "benchmark_v2_results.csv"
        manifest_path = run_dir / "manifest.json"
        self._write_manifest(manifest_path, conditions, num_trials)

        print(f"Benchmark v2 output: {run_dir}")
        print(f"Conditions: {len(conditions)} | trials per condition: {num_trials}")

        all_results: list[dict[str, Any]] = []
        for idx, condition in enumerate(conditions, start=1):
            print(f"[{idx}/{len(conditions)}] {condition.name}")
            for trial in range(num_trials):
                all_results.append(self.run_trial(condition, trial))
            pd.DataFrame(all_results).to_csv(csv_path, index=False)
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        return run_dir

    def run_trial(self, condition: BenchmarkV2Condition, trial: int) -> dict[str, Any]:
        seed = self.base_seed + trial
        rng = random.Random(seed + condition.grid_size * 1000 + condition.npc_count)
        npc_starts, player_start = spread_start_positions(
            rng=rng,
            npc_count=condition.npc_count,
            world_size=condition.grid_size,
        )
        target_color, target_shape = "red", "triangle"
        target_label = f"{target_color}_{target_shape}"
        exploration_ticks = exploration_ticks_for(condition.grid_size, condition.npc_count)

        overrides = {
            "GRID_SIZE": condition.grid_size,
            "NPC_COUNT": condition.npc_count,
            "NPC_START": npc_starts[0],
            "NPC_STARTS": npc_starts if condition.npc_count > 1 else None,
            "NPC_SIGHT_RANGE": BENCHMARK_NPC_SIGHT_RANGE,
            "PLAYER_START": player_start,
            "PLAYER_SIGHT_RANGE": BENCHMARK_PLAYER_SIGHT_RANGE,
            "NPC_EXPLORATION_TICKS": exploration_ticks,
            "NPC_GOAL": True,
            "NPC_GOAL_DETERMINISTIC": True,
            "NPC_GOAL_COLOR": "blue",
            "NPC_GOAL_SHAPE": "circle",
            "NPC_KNOWLEDGE_MODE": condition.knowledge_mode,
            "NPC_RESPONSE_MODE": condition.response_mode,
            "NPC_COMPETING": condition.competitor_count > 0,
            "NPC_COMPETING_COUNT": condition.competitor_count,
            "NPC_MULTIPLE_KNOWLEDGE_MODE": "independent",
            "NPC_NONCOMPETING_GOAL_MODE": "shared",
            "NPC_NPC_INTERACTION_ENABLED": True,
            "NPC_MEMORY_DECAY_TICKS": None,
            "NPC_SELECTIVE_ATTENTION": None,
            "NPC_SLM_REGION_GROUNDING": False,
            "NPC_USE_LLM_JUDGE": self.use_judge,
            "RANDOM_SPAWN": True,
            "WORLD_RESERVED_POSITIONS": benchmark_reserved_positions(condition.grid_size),
        }

        with temporary_config(**overrides):
            world = GameWorld(target_color=target_color, target_shape=target_shape, seed=seed)
            player = Player(*player_start, sight_range=config.PLAYER_SIGHT_RANGE)
            world.update_player_vision(player)
            party = NPCParty.from_config(world, target_label)
            speaker = party.actors[0]
            target_location = self._find_target_location(world, target_label)
            tracker = TargetAvailabilityTracker(target_label, target_location)
            tracker.record(party, tick=0)
            interaction_manager = InteractionManager(
                api=PygameGameAPI(world, player, party.brain_map),
                slm_client=self._get_slm_client() if condition.response_mode == "slm" else None,
                enforce_grounding=False,
            )

            tag = self._condition_tag(condition, trial)
            logger = GameLogger.start(
                world,
                player,
                party,
                tag=tag,
                seed=seed,
                extra_meta={
                    "benchmark": "v2",
                    "condition": asdict(condition),
                    "trial": trial,
                    "npc_starts": [list(pos) for pos in npc_starts],
                    "player_start": list(player_start),
                    "exploration_ticks": exploration_ticks,
                },
            )

            if condition.knowledge_mode == "embodied":
                for tick in range(1, config.NPC_EXPLORATION_TICKS + 1):
                    tick_result = party.tick()
                    tracker.record(party, tick=tick)
                    logger.log_tick(
                        world,
                        player,
                        party,
                        event_msgs=tick_result.event_msgs,
                        knowledge_exchanges=tick_result.knowledge_exchanges,
                    )

            availability = tracker.summary(party, speaker.npc_id)
            speaker_memory = dict(speaker.brain.state.shape_locations)
            party_memory = self._combined_shape_locations(party)
            question = self._create_natural_question(target_label)
            interaction_id = logger.log_interaction_pre(
                world,
                player,
                speaker,
                target_color,
                target_shape,
            )
            start_time = time.perf_counter()
            _, response_text = interaction_manager.start_interaction(
                speaker.brain,
                target_color,
                target_shape,
            )
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.log_interaction_summary(
                interaction_id,
                interaction_manager,
                speaker,
                world,
                target_color,
                target_shape,
                question,
                response_text,
            )

            speaker_metrics = self._evaluate_response(
                response_text,
                speaker_memory,
                availability["speaker_target_available"],
                target_label,
                target_location,
                world.size,
                tool_calls=interaction_manager.last_tool_calls,
            )
            party_metrics = self._evaluate_response(
                response_text,
                party_memory,
                availability["party_target_available"],
                target_label,
                target_location,
                world.size,
                tool_calls=interaction_manager.last_tool_calls,
            )

            logger.end(
                outcome="benchmark_v2_complete",
                extra_stats={
                    "condition": condition.name,
                    "speaker_outcome_bucket": speaker_metrics["outcome_bucket"],
                    "party_outcome_bucket": party_metrics["outcome_bucket"],
                    "response_time_ms": response_time_ms,
                    **availability,
                },
            )

            return {
                "benchmark": "v2",
                "trial": trial,
                "seed": seed,
                "condition": condition.name,
                "scope": condition.scope,
                "grid_size": condition.grid_size,
                "exploration_ticks": exploration_ticks,
                "npc_sight_range": BENCHMARK_NPC_SIGHT_RANGE,
                "player_sight_range": BENCHMARK_PLAYER_SIGHT_RANGE,
                "slm_region_grounding": False,
                "llm_judge_enabled": self.use_judge,
                "npc_count": condition.npc_count,
                "knowledge_mode": condition.knowledge_mode,
                "response_mode": condition.response_mode,
                "competitor_mode": condition.competitor_mode,
                "competitor_count": condition.competitor_count,
                "npc_starts": npc_starts,
                "player_start": player_start,
                "speaker_npc_id": speaker.npc_id,
                "question": question,
                "response_text": response_text,
                "raw_response_text": interaction_manager.last_raw_response,
                "final_response_text": interaction_manager.last_response,
                "tool_calls": interaction_manager.last_tool_calls,
                "grounding_violation": interaction_manager.last_grounding_violation,
                "grounding_violations": interaction_manager.last_grounding_violations,
                "token_usage": interaction_manager.last_token_usage,
                "token_usage_total": interaction_manager.last_token_usage_total,
                "llm_error": interaction_manager.last_llm_error,
                "response_time_ms": response_time_ms,
                "target_location": target_location,
                "run_id": logger.run_id,
                "speaker_npc_steps": speaker.npc.steps_taken,
                "speaker_npc_coverage": speaker.brain.state.coverage,
                "speaker_direct_coverage": speaker.brain.state.direct_coverage,
                "party_combined_coverage": self._combined_direct_coverage(party),
                "party_combined_direct_coverage": self._combined_direct_coverage(party),
                "party_combined_knowledge_coverage": self._combined_knowledge_coverage(party),
                **availability,
                **{f"speaker_{key}": value for key, value in speaker_metrics.items()},
                **{f"party_{key}": value for key, value in party_metrics.items()},
            }

    def _create_output_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = self.output_root / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = self.output_root / f"{timestamp}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _write_manifest(
        self,
        path: Path,
        conditions: list[BenchmarkV2Condition],
        num_trials: int,
    ) -> None:
        manifest = {
            "benchmark": "v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_trials": num_trials,
            "base_seed": self.base_seed,
            "use_judge": self.use_judge,
            "conditions": [asdict(condition) for condition in conditions],
            "config_defaults": self._config_snapshot(),
            "effective_benchmark_settings": self._effective_benchmark_settings(),
            "exploration_ticks_by_grid_and_npc_count": [
                {"grid_size": grid_size, "npc_count": npc_count, "ticks": ticks}
                for (grid_size, npc_count), ticks
                in sorted(BENCHMARK_EXPLORATION_TICKS_BY_GRID_AND_NPCS.items())
            ],
            "git": self._git_snapshot(),
            "notes": {
                "slm_region_grounding": "disabled for raw SLM behavior",
                "memory_decay": "not included in v2 matrix",
                "selective_attention": "not included in v2 matrix",
                "party_knowledge": "independent only; NPCs can exchange memory locally",
                "reserved_positions": (
                    "target/object layout reserves stable anchors so paired seeds "
                    "keep shape placement comparable across NPC counts"
                ),
            },
        }
        path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _git_snapshot() -> dict[str, Any]:
        def run(args: list[str]) -> str:
            try:
                return subprocess.check_output(args, text=True).strip()
            except Exception as exc:
                return f"unavailable: {exc}"

        return {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "branch", "--show-current"]),
            "status_short": run(["git", "status", "--short"]),
        }

    @staticmethod
    def _config_snapshot() -> dict[str, Any]:
        keys = [
            "GRID_SIZE",
            "NUM_OBJECTS",
            "RANDOM_SPAWN",
            "NPC_COUNT",
            "NPC_SIGHT_RANGE",
            "NPC_EXPLORATION_TICKS",
            "PLAYER_SIGHT_RANGE",
            "NPC_SLM_MODEL_ID",
            "NPC_SLM_ENABLE_TOOL_CALLS",
            "NPC_SLM_USE_CHAT_TEMPLATE",
            "NPC_SLM_INCLUDE_COORDS",
            "NPC_SLM_TOOL_WHITELIST",
            "NPC_SLM_REGION_GROUNDING",
            "NPC_USE_LLM_JUDGE",
            "NPC_JUDGE_MODEL",
            "DEFAULT_GEMINI_MODEL",
        ]
        return {key: getattr(config, key, None) for key in keys}

    @staticmethod
    def _effective_benchmark_settings() -> dict[str, Any]:
        return {
            "grid_sizes": DEFAULT_GRID_SIZES,
            "party_sizes": DEFAULT_PARTY_SIZES,
            "response_modes": DEFAULT_RESPONSE_MODES,
            "num_trials": DEFAULT_NUM_TRIALS,
            "npc_sight_range": BENCHMARK_NPC_SIGHT_RANGE,
            "player_sight_range": BENCHMARK_PLAYER_SIGHT_RANGE,
            "target_color": "red",
            "target_shape": "triangle",
            "npc_goal_enabled": True,
            "npc_goal_deterministic": True,
            "npc_goal_color": "blue",
            "npc_goal_shape": "circle",
            "multiple_knowledge_mode": "independent",
            "noncompeting_goal_mode": "shared",
            "npc_interaction_enabled": True,
            "memory_decay_ticks": None,
            "selective_attention": None,
            "slm_region_grounding": False,
            "random_spawn": True,
            "reserved_object_positions": "stable benchmark anchors",
        }

    @staticmethod
    def _condition_tag(condition: BenchmarkV2Condition, trial: int) -> str:
        pieces = [
            "benchmarkv2",
            condition.scope,
            f"n{condition.npc_count}",
            f"g{condition.grid_size}",
            condition.knowledge_mode,
            condition.response_mode,
        ]
        if condition.competitor_mode != "none":
            pieces.append(f"comp-{condition.competitor_mode}")
        pieces.append(f"trial{trial}")
        return "_".join(pieces)

    @staticmethod
    def _create_natural_question(target_label: str) -> str:
        return f"Where is the {get_natural_object_name(target_label)}?"

    @staticmethod
    def _find_target_location(
        world: GameWorld,
        target_label: str,
    ) -> tuple[int, int] | None:
        for shape in world.shapes:
            if shape.label == target_label:
                return (shape.x, shape.y)
        return None

    @staticmethod
    def _combined_shape_locations(party: NPCParty) -> dict[str, list[tuple[int, int]]]:
        combined: dict[str, set[tuple[int, int]]] = {}
        for actor in party.actors:
            for label, positions in actor.brain.state.shape_locations.items():
                combined.setdefault(label, set()).update(positions)
        return {label: sorted(positions) for label, positions in combined.items()}

    @staticmethod
    def _combined_direct_coverage(party: NPCParty) -> float:
        cells: set[tuple[int, int]] = set()
        for actor in party.actors:
            cells |= set(actor.brain.state.direct_observed_cells)
        return len(cells) / (party.world.size ** 2)

    @staticmethod
    def _combined_knowledge_coverage(party: NPCParty) -> float:
        cells: set[tuple[int, int]] = set()
        for actor in party.actors:
            cells |= set(actor.brain.state.observed_cells)
        return len(cells) / (party.world.size ** 2)

    @staticmethod
    def _evaluate_response(
        response: str,
        observed_memory: dict[str, list[tuple[int, int]]],
        target_available: bool,
        target_label: str,
        target_location: tuple[int, int] | None,
        world_size: int,
        *,
        tool_calls: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        outcome = metrics.classify_outcome(
            response,
            target_label,
            target_available,
            target_location,
            tool_calls=tool_calls,
            world_size=world_size,
        )
        grounded = metrics.score_groundedness(response, target_label, observed_memory)
        relevance = metrics.score_relevance(response, target_label)
        return {
            "outcome_bucket": outcome["outcome_bucket"],
            "correct_via": outcome.get("correct_via"),
            "chebyshev_distance": outcome["chebyshev_distance"],
            "had_mixed_content": outcome["had_mixed_content"],
            "groundedness_rate": grounded["rate"],
            "n_claims": grounded["n_claims"],
            "n_grounded": grounded["n_grounded"],
            "n_shape_confusion": grounded["n_shape_confusion"],
            "n_fabricated": grounded["n_fabricated"],
            "on_topic": relevance["on_topic"],
            "committal": relevance["committal"],
            "false_refusal": outcome["outcome_bucket"] == "false_refusal",
            "correct_behavior": outcome["outcome_bucket"] in {"correct", "correct_abstention"},
        }


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv_arg(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_arg(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark v2.")
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--grid-sizes", default=",".join(map(str, DEFAULT_GRID_SIZES)))
    parser.add_argument("--party-sizes", default=",".join(map(str, DEFAULT_PARTY_SIZES)))
    parser.add_argument("--response-modes", default=",".join(DEFAULT_RESPONSE_MODES))
    parser.add_argument("--use-judge", action="store_true")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="Print the condition matrix and exit without running trials.",
    )
    args = parser.parse_args()

    conditions = build_conditions(
        grid_sizes=_parse_int_csv_arg(args.grid_sizes),
        party_sizes=_parse_int_csv_arg(args.party_sizes),
        response_modes=_parse_csv_arg(args.response_modes),
    )
    if args.list_conditions:
        for condition in conditions:
            print(condition.name)
        print(f"total conditions: {len(conditions)}")
        return

    runner = BenchmarkV2Runner(
        output_root=Path(args.output_root),
        use_judge=args.use_judge,
    )
    run_dir = runner.run(conditions=conditions, num_trials=args.num_trials)
    print(f"Benchmark v2 complete: {run_dir}")


if __name__ == "__main__":
    main()
