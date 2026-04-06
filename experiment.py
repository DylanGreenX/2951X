"""
Experimental framework — runs test configurations and collects data.

Behavior logic lives in domain objects (npc_brain.py, interaction.py).
Analysis lives in analyze_results.py.
"""

import time
import re
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any

import config
from world import GameWorld
from entities import Player, NPC
from npc_brain import NPCBrainGoalDriven, NPCBrainWandering
from interaction import InteractionManager


@dataclass
class ExperimentCondition:
    name: str
    knowledge_mode: str  # "perfect" | "embodied"
    response_mode: str   # "deterministic" | "llm" | "slm"


class ExperimentRunner:

    def __init__(self):
        self.interaction_manager = InteractionManager()

    def run_condition(self, condition: ExperimentCondition, num_trials: int = 50) -> List[Dict[str, Any]]:
        print(f"Running {condition.name} ({num_trials} trials)...")
        results = []
        for trial in range(num_trials):
            results.append(self._run_trial(condition, trial))
            if (trial + 1) % 10 == 0:
                print(f"  Completed {trial + 1}/{num_trials} trials")
        return results

    def _run_trial(self, condition: ExperimentCondition, trial: int) -> Dict[str, Any]:
        world, player, npc, brain = self._init_trial(trial, condition.knowledge_mode)

        if condition.knowledge_mode == "embodied":
            for _ in range(150):
                brain.tick()

        npc_knowledge = brain.state.to_llm_context().copy()
        target_label = f"{world.target_color}_{world.target_shape}"
        target_location = self._find_target_location(world, target_label)

        original_mode = config.NPC_RESPONSE_MODE
        config.NPC_RESPONSE_MODE = condition.response_mode
        try:
            question = f"Where is the {world.target_color} {world.target_shape}?"
            start_time = time.perf_counter()
            _, response_text = self.interaction_manager.start_interaction(
                brain, world.target_color, world.target_shape
            )
            response_time_ms = (time.perf_counter() - start_time) * 1000
        finally:
            config.NPC_RESPONSE_MODE = original_mode

        metrics = self._evaluate_response(response_text, target_location, npc_knowledge)

        return {
            'trial': trial,
            'condition': condition.name,
            'question': question,
            'response_text': response_text,
            'response_time_ms': response_time_ms,
            'npc_steps': npc.steps_taken,
            'npc_coverage': brain.state.coverage,
            'npc_knowledge': npc_knowledge,
            'target_location': target_location,
            'target_was_observed': brain.state.seen_label(target_label),
            'knowledge_mode': condition.knowledge_mode,
            'response_mode': condition.response_mode,
            **metrics
        }

    def _init_trial(self, seed: int, knowledge_mode: str):
        world = GameWorld(target_color="red", target_shape="triangle", seed=seed)
        player = Player(*config.PLAYER_START, sight_range=config.PLAYER_SIGHT_RANGE)
        npc = NPC(*config.NPC_START, sight_range=config.NPC_SIGHT_RANGE)
        brain = NPCBrainGoalDriven(npc, world, goal_label="blue_circle")
        if knowledge_mode == "perfect":
            self._give_perfect_knowledge(brain, world)
        return world, player, npc, brain

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

    def _evaluate_response(self, response: str, target_location: tuple, npc_knowledge: List[str]) -> Dict[str, float]:
        return {
            'accuracy': self._eval_accuracy(response, target_location),
            'relevance': self._eval_relevance(response, target_location),
            'groundedness': self._eval_groundedness(response, npc_knowledge),
        }

    def _eval_accuracy(self, response: str, target_location: tuple) -> float:
        if target_location is None:
            return 1.0 if "haven't seen" in response.lower() or "don't know" in response.lower() else 0.0
        coord_str = f"({target_location[0]}, {target_location[1]})"
        return 1.0 if coord_str in response else 0.0

    def _eval_relevance(self, response: str, target_location: tuple) -> float:
        if target_location is None:
            return 1.0 if "haven't" in response.lower() else 0.0
        if f"({target_location[0]}, {target_location[1]})" in response:
            return 1.0
        elif any(word in response.lower() for word in ["north", "south", "east", "west", "near", "area"]):
            return 0.5
        elif "seen" in response.lower():
            return 0.3
        return 0.0

    def _eval_groundedness(self, response: str, npc_knowledge: List[str]) -> float:
        response_coords = re.findall(r'\((\d+),\s*(\d+)\)', response)
        knowledge_text = " ".join(npc_knowledge)
        knowledge_coords = re.findall(r'\((\d+),\s*(\d+)\)', knowledge_text)
        if not response_coords:
            return 1.0
        grounded = sum(1 for c in response_coords if c in knowledge_coords)
        return grounded / len(response_coords)


CORE_CONDITIONS = [
    ExperimentCondition("Perfect + Deterministic", "perfect", "deterministic"),
    ExperimentCondition("Perfect + LLM", "perfect", "llm"),
    ExperimentCondition("Perfect + SLM", "perfect", "slm"),
    ExperimentCondition("Embodied + Deterministic", "embodied", "deterministic"),
    ExperimentCondition("Embodied + LLM", "embodied", "llm"),
    ExperimentCondition("Embodied + SLM", "embodied", "slm"),
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
    print(results[['condition', 'response_time_ms', 'accuracy', 'target_was_observed']].head())
