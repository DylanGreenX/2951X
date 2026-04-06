"""
Interaction system between player and NPC.

Handles proximity detection, the deterministic question/response cycle,
and provides a stub for future LLM/SLM integration.

The deterministic response is a direct lookup against the NPC's RLangState —
no language model involved. This is the baseline condition for benchmarking.
"""
from entities import Player, NPC
from npc_brain import NPCBrain
import config


class InteractionManager:
    def can_interact(self, player: Player, npc: NPC) -> bool:
        """Player must be on the same cell as the NPC to interact."""
        return player.x == npc.x and player.y == npc.y

    def start_interaction(self, brain, target_color, target_shape):
        question = f"Where is the {target_color} {target_shape}?"
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
        """
        Look up the target in the NPC's observed shape locations and return
        a structured natural language response.

        This is the baseline deterministic NPC condition — no LLM involved.
        The NPC reports only what it has personally observed.
        """
        label = f"{target_color}_{target_shape}"
        locations = brain.state.shape_locations.get(label)

        if not locations:
            return (
                f"I haven't seen a {target_color} {target_shape} in my travels. "
                f"I've only explored {brain.state.coverage:.0%} of the world so far."
            )

        pos_str = ", ".join(f"({x}, {y})" for x, y in locations)
        count = len(locations)
        if count == 1:
            return f"Yes! I saw a {target_color} {target_shape} at {pos_str}."
        return f"I've seen {count} {target_color} {target_shape}(s) at: {pos_str}."

    def get_llm_response(self, brain: NPCBrain, target_color: str, target_shape: str, model: str) -> str:
        """
        Build prompt from NPC's RLang knowledge and send to language model.
        Competitive brains modify context via get_sharing_context().
        TODO(Marcus): Replace _call_llm/_call_slm with API layer calls.
        """
        if hasattr(brain, 'get_sharing_context'):
            context_lines = brain.get_sharing_context(target_color, target_shape)
        else:
            context_lines = brain.state.to_llm_context()

        context_str = "\n".join(context_lines)
        question = f"Where is the {target_color} {target_shape}?"

        prompt = (
            "You are a helpful NPC in a grid world. You can ONLY share "
            "information you have personally observed. Never make up information.\n\n"
            f"Current knowledge:\n{context_str}\n\n"
            f"The player is looking for a {target_color} {target_shape}. "
            "Respond in 1-2 sentences based only on what you know.\n\n"
            f'Player: "{question}"\nNPC:'
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
