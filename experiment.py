"""
Experimental framework — runs test configurations and collects data.

Behavior logic lives in domain objects (npc_brain.py, interaction.py).
Analysis lives in analyze_results.py.
"""

import time
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any

import config
from world import GameWorld
from entities import Player, NPC
from npc_brain import NPCBrainGoalDriven, NPCBrainWandering
from interaction import InteractionManager
from llm import SLMClient
from pygame_game_api import PygameGameAPI
from rlang_engine import get_natural_object_name
from game_log import GameLogger
import metrics
import judge


@dataclass
class _SavedConfig:
    """Snapshot/restore for the handful of config globals an ExperimentCondition
    can override. Keeps the trial body free of save/restore boilerplate."""
    response_mode: str
    knowledge_mode: str
    competing: bool
    selective_attention: str | None
    memory_decay_ticks: int | None

    @classmethod
    def snapshot(cls) -> "_SavedConfig":
        return cls(
            response_mode=config.NPC_RESPONSE_MODE,
            knowledge_mode=config.NPC_KNOWLEDGE_MODE,
            competing=config.NPC_COMPETING,
            selective_attention=config.NPC_SELECTIVE_ATTENTION,
            memory_decay_ticks=config.NPC_MEMORY_DECAY_TICKS,
        )

    def restore(self) -> None:
        config.NPC_RESPONSE_MODE = self.response_mode
        config.NPC_KNOWLEDGE_MODE = self.knowledge_mode
        config.NPC_COMPETING = self.competing
        config.NPC_SELECTIVE_ATTENTION = self.selective_attention
        config.NPC_MEMORY_DECAY_TICKS = self.memory_decay_ticks


@dataclass
class ExperimentCondition:
    name: str
    knowledge_mode: str  # "perfect" | "embodied"
    response_mode: str   # "deterministic" | "llm" | "slm"

    # Extended modality overrides. All default to "inherit from config"; when
    # set on a condition, ExperimentRunner._run_trial applies the override for
    # that trial only. See README "Configuring experiments" for semantics.
    competing: bool = False
    selective_attention: str | None = None  # "color" | "shape" | None
    memory_decay_ticks: int | None = None


class ExperimentRunner:
    def __init__(self) -> None:
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

    def run_condition(self, condition: ExperimentCondition, num_trials: int = 50) -> List[Dict[str, Any]]:
        print(f"Running {condition.name} ({num_trials} trials)...")
        slm_client = self._get_slm_client() if condition.response_mode == "slm" else None
        results = []
        for trial in range(num_trials):
            results.append(self._run_trial(condition, trial, slm_client=slm_client))
            if (trial + 1) % 10 == 0:
                print(f"  Completed {trial + 1}/{num_trials} trials")
        return results

    def _run_trial(
        self,
        condition: ExperimentCondition,
        trial: int,
        slm_client: SLMClient | None = None,
    ) -> Dict[str, Any]:
        # Modality flags have to be applied BEFORE init_trial because the goal
        # label in competitive mode depends on NPC_COMPETING, and selective
        # attention / decay are read inside RLangState.observe during the
        # exploration warmup below.
        saved = _SavedConfig.snapshot()
        config.NPC_RESPONSE_MODE = condition.response_mode
        config.NPC_KNOWLEDGE_MODE = condition.knowledge_mode
        config.NPC_COMPETING = condition.competing
        config.NPC_SELECTIVE_ATTENTION = condition.selective_attention
        config.NPC_MEMORY_DECAY_TICKS = condition.memory_decay_ticks

        try:
            world, player, npc, brain = self._init_trial(trial, condition.knowledge_mode, condition)

            # Tag encodes every axis so run_ids are self-describing on disk.
            tag_parts = [
                "experiment",
                condition.knowledge_mode,
                condition.response_mode,
            ]
            if condition.competing:
                tag_parts.append("competing")
            if condition.selective_attention:
                tag_parts.append(f"attn-{condition.selective_attention}")
            if condition.memory_decay_ticks is not None:
                tag_parts.append(f"decay-{condition.memory_decay_ticks}")
            tag_parts.append(f"trial{trial}")
            tag = "_".join(tag_parts)

            logger = GameLogger.start(
                world, player, npc, brain,
                tag=tag, seed=trial,
                extra_meta={
                    "condition_name": condition.name,
                    "knowledge_mode": condition.knowledge_mode,
                    "response_mode": condition.response_mode,
                    "competing": condition.competing,
                    "selective_attention": condition.selective_attention,
                    "memory_decay_ticks": condition.memory_decay_ticks,
                    "trial": trial,
                },
            )

            if condition.knowledge_mode == "embodied":
                for _ in range(config.NPC_EXPLORATION_TICKS):
                    brain.tick()
                    logger.log_tick(world, player, npc, brain)

            npc_knowledge = brain.state.to_llm_context().copy()
            target_label = f"{world.target_color}_{world.target_shape}"
            target_location = self._find_target_location(world, target_label)
            interaction_manager = InteractionManager(
                api=PygameGameAPI.from_game(world, player, brain),
                slm_client=slm_client,
                enforce_grounding=False,
            )

            question = self._create_natural_question(world.target_color, world.target_shape)
            interaction_id = logger.log_interaction_pre(
                world, player, npc, brain, world.target_color, world.target_shape
            )
            start_time = time.perf_counter()
            _, response_text = interaction_manager.start_interaction(
                brain, world.target_color, world.target_shape
            )
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.log_interaction_summary(
                interaction_id, interaction_manager, brain, world,
                world.target_color, world.target_shape,
                question, response_text,
            )
        finally:
            saved.restore()

        metrics_dict = self._evaluate_response(
            response_text, brain, world, target_label, target_location,
            tool_calls=interaction_manager.last_tool_calls,
        )

        logger.end(
            outcome="experiment_complete",
            extra_stats={
                "npc_steps": npc.steps_taken,
                "npc_coverage": round(brain.state.coverage, 4),
                "target_was_observed": brain.state.seen_label(target_label),
                "outcome_bucket": metrics_dict.get("outcome_bucket"),
                "response_time_ms": response_time_ms,
            },
        )

        return {
            'trial': trial,
            'condition': condition.name,
            'question': question,
            'response_text': response_text,
            'raw_response_text': interaction_manager.last_raw_response,
            'final_response_text': interaction_manager.last_response,
            'tool_calls': interaction_manager.last_tool_calls,
            'grounding_violation': interaction_manager.last_grounding_violation,
            'grounding_violations': interaction_manager.last_grounding_violations,
            'token_usage': interaction_manager.last_token_usage,
            'token_usage_total': interaction_manager.last_token_usage_total,
            'llm_error': interaction_manager.last_llm_error,
            'response_time_ms': response_time_ms,
            'npc_steps': npc.steps_taken,
            'npc_coverage': brain.state.coverage,
            'npc_knowledge': npc_knowledge,
            'target_location': target_location,
            'target_was_observed': brain.state.seen_label(target_label),
            'knowledge_mode': condition.knowledge_mode,
            'response_mode': condition.response_mode,
            'competing': condition.competing,
            'selective_attention': condition.selective_attention,
            'memory_decay_ticks': condition.memory_decay_ticks,
            'run_id': logger.run_id,
            **metrics_dict
        }

    def _init_trial(
        self,
        seed: int,
        knowledge_mode: str,
        condition: ExperimentCondition | None = None,
    ):
        world = GameWorld(target_color="red", target_shape="triangle", seed=seed)
        player = Player(*config.PLAYER_START, sight_range=config.PLAYER_SIGHT_RANGE)
        npc = NPC(*config.NPC_START, sight_range=config.NPC_SIGHT_RANGE)

        # Competitive mode: NPC chases the same item the player is looking for,
        # so its observation path is biased toward the target. The sharing
        # policy in interaction.py separately handles withholding.
        if condition and condition.competing:
            goal_label = f"{world.target_color}_{world.target_shape}"
        else:
            goal_label = "blue_circle"
        brain = NPCBrainGoalDriven(npc, world, goal_label=goal_label)

        if knowledge_mode == "perfect":
            self._give_perfect_knowledge(brain, world)
        return world, player, npc, brain

    def _create_natural_question(self, target_color: str, target_shape: str) -> str:
        label = f"{target_color}_{target_shape}"
        return f"Please show me where the {get_natural_object_name(label)} is."

    def _give_perfect_knowledge(self, brain, world):
        for x in range(world.size):
            for y in range(world.size):
                brain.state.observed_cells.add((x, y))
        for shape in world.shapes:
            brain.state.observed_shapes.append(shape)
            brain.state.shape_locations.setdefault(shape.label, [])
            brain.state.shape_locations[shape.label].append((shape.x, shape.y))

    def _find_target_location(self, world, target_label: str):
        for shape in world.shapes:
            if shape.label == target_label:
                return (shape.x, shape.y)
        return None

    def _evaluate_response(
        self,
        response: str,
        brain,
        world,
        target_label: str,
        target_location: tuple | None,
        *,
        tool_calls: list | None = None,
    ) -> Dict[str, Any]:
        """
        Structural, claim-level scoring stratified by whether the NPC actually
        observed the target. Accepts the tool-call trace so correctness can
        resolve via set_npc_target action in addition to coord / region text.
        See metrics.py for bucket definitions.

        Naturalness is scored only by the LLM judge — regex cannot evaluate
        style — and is surfaced as a separate axis per the 4-way analysis
        (accuracy × naturalness × knowledge × response-mode).
        """
        observed = brain.state.shape_locations
        target_was_observed = brain.state.seen_label(target_label)

        outcome = metrics.classify_outcome(
            response, target_label, target_was_observed, target_location,
            tool_calls=tool_calls, world_size=world.size,
        )
        grounded = metrics.score_groundedness(response, target_label, observed)
        relevance = metrics.score_relevance(response, target_label)

        bucket = outcome["outcome_bucket"]
        result: Dict[str, Any] = {
            # Regex scoring — the fast path
            "outcome_bucket": bucket,
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
            "false_refusal": bucket == "false_refusal",
        }

        # Optional LLM judge — dual-logged with regex so we can compute
        # per-bucket agreement and surface the cases where they disagree.
        # Also produces the naturalness score (regex cannot do style).
        if getattr(config, "NPC_USE_LLM_JUDGE", False):
            j = judge.classify(
                response, target_label, target_location,
                target_was_observed, observed,
                model=getattr(config, "NPC_JUDGE_MODEL", "gemini-2.5-flash"),
                tool_calls=tool_calls,
                world_size=world.size,
            )
            j_bucket = j.get("outcome_bucket", "judge_error")
            result.update({
                "judge_bucket":          j_bucket,
                "judge_correct_via":     j.get("correct_via"),
                "judge_chebyshev":       j.get("chebyshev_distance"),
                "judge_had_mixed":       j.get("had_mixed_content"),
                "judge_on_topic":        j.get("on_topic"),
                "judge_committal":       j.get("committal"),
                "judge_n_claims":        j.get("n_claims"),
                "judge_n_grounded":      j.get("n_grounded"),
                "judge_n_shape_conf":    j.get("n_shape_confusion"),
                "judge_n_fabricated":    j.get("n_fabricated"),
                "judge_groundedness":    j.get("groundedness_rate"),
                "judge_naturalness":     j.get("naturalness"),
                "judge_reasoning":       j.get("reasoning"),
                "judge_error":           j.get("judge_error"),
                "regex_judge_agree":     j_bucket == bucket,
            })
        return result


CORE_CONDITIONS = [
    ExperimentCondition("Perfect + Deterministic", "perfect", "deterministic"),
    ExperimentCondition("Perfect + LLM", "perfect", "llm"),
    ExperimentCondition("Perfect + SLM", "perfect", "slm"),
    ExperimentCondition("Embodied + Deterministic", "embodied", "deterministic"),
    ExperimentCondition("Embodied + LLM", "embodied", "llm"),
    ExperimentCondition("Embodied + SLM", "embodied", "slm"),
]


# Extended modalities — LLM-only by design. Each modality changes either what
# the NPC knows when asked or what it chooses to reveal; running deterministic
# responses across them would be lookup-table variants and miss the point.
EXTENDED_CONDITIONS = [
    ExperimentCondition(
        "Embodied + LLM + Competitive", "embodied", "llm",
        competing=True,
    ),
    ExperimentCondition(
        "Embodied + LLM + MemoryDecay", "embodied", "llm",
        memory_decay_ticks=20,
    ),
    ExperimentCondition(
        "Embodied + LLM + SelectiveAttention", "embodied", "llm",
        selective_attention="color",
    ),
]


def run_core_experiments(num_trials: int = 50, response_filter: str = None) -> pd.DataFrame:
    """
    Run core experiment matrix. Pass response_filter="deterministic" to run
    only deterministic conditions, or None for the full 2x3 matrix.
    """
    runner = ExperimentRunner()
    conditions = CORE_CONDITIONS
    if response_filter:
        conditions = [c for c in conditions if c.response_mode == response_filter]

    all_results = []
    for condition in conditions:
        all_results.extend(runner.run_condition(condition, num_trials))

    df = pd.DataFrame(all_results)
    df.to_csv('core_experiments_results.csv', index=False)
    print(f"Results saved to core_experiments_results.csv")
    return df


if __name__ == "__main__":
    results = run_core_experiments(num_trials=5, response_filter="deterministic")
    print("Sample results:")
    print(results[['condition', 'response_time_ms', 'outcome_bucket', 'target_was_observed']].head())
