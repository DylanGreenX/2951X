"""
Interaction system between player and NPC.

Handles proximity detection, the deterministic question/response cycle,
and provides a stub for future LLM/SLM integration.

The deterministic response is a direct lookup against the NPC's RLangState —
no language model involved. This is the baseline condition for benchmarking.
"""
from entities import Player, NPC
from npc_brain import NPCBrain
from rlang_engine import get_natural_object_name, get_natural_location_name
import config


class InteractionManager:
    def can_interact(self, player: Player, npc: NPC) -> bool:
        """Player must be on the same cell as the NPC to interact."""
        return player.x == npc.x and player.y == npc.y

    def start_interaction(self, brain, target_color, target_shape):
        label = f"{target_color}_{target_shape}"
        natural_name = get_natural_object_name(label)
        question = f"Where is the {natural_name}?"

        if config.NPC_RESPONSE_MODE == "deterministic":
            response = self.get_deterministic_response(brain, target_color, target_shape)
        elif config.NPC_RESPONSE_MODE in ("llm", "slm"):
            response = self.get_llm_response(brain, target_color, target_shape, model=config.NPC_RESPONSE_MODE)
        else:
            raise ValueError(f"Unknown NPC_RESPONSE_MODE: {config.NPC_RESPONSE_MODE}")
        return question, response

    def get_deterministic_response(
        self,
        brain: NPCBrain,
        target_color: str,
        target_shape: str,
    ) -> str:
        """Baseline deterministic lookup — no LLM involved."""
        label = f"{target_color}_{target_shape}"
        natural_name = get_natural_object_name(label)
        locations = brain.state.shape_locations.get(label)

        if not locations:
            return (
                f"I haven't seen any {natural_name} in my travels. "
                f"I've only explored {brain.state.coverage:.0%} of this region so far."
            )

        natural_locations = [get_natural_location_name(x, y) for x, y in locations]
        count = len(locations)
        if count == 1:
            return f"Aye, I found a {natural_name} {natural_locations[0]}."

        location_list = ", ".join(natural_locations)
        return f"I've seen {count} {natural_name}s at: {location_list}."

    def get_llm_response(self, brain: NPCBrain, target_color: str, target_shape: str, model: str) -> str:
        """Build prompt from NPC's RLang knowledge. TODO(Marcus): wire up API layer."""
        if hasattr(brain, 'get_sharing_context'):
            context_lines = brain.get_sharing_context(target_color, target_shape)
        else:
            context_lines = brain.state.to_llm_context()

        context_str = "\n".join(context_lines)

        label = f"{target_color}_{target_shape}"
        natural_name = get_natural_object_name(label)
        question = f"Where is the {natural_name}?"

        prompt = (
            "You are a helpful traveler in Skyrim. You can ONLY share "
            "information about items and locations you have personally seen during your travels. "
            "Never make up information or guess about locations.\n\n"
            f"Your observations:\n{context_str}\n\n"
            f"A fellow traveler is looking for a {natural_name}. "
            "Respond naturally in 1-2 sentences based only on what you've witnessed.\n\n"
            f'Traveler: "{question}"\nYou:'
        )

        if model == "llm":
            return self._call_llm(prompt)
        elif model == "slm":
            return self._call_slm(prompt)
        raise ValueError(f"Unknown model: {model}")

    def _call_llm(self, prompt: str) -> str:
        # TODO(Marcus): Implement with full LLM API call
        raise NotImplementedError("LLM integration not yet implemented.")

    def _call_slm(self, prompt: str) -> str:
        # TODO(Gordon): Implement with SLM (TinyLlama, Phi, etc.)
        raise NotImplementedError("SLM integration not yet implemented.")
