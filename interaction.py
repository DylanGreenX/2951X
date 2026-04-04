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

    def get_llm_response(
        self,
        brain: NPCBrain,
        target_color: str,
        target_shape: str,
        model: str,
    ) -> str:
        # TODO(Marcus): Replace with API layer call.
        # Input:  brain.state.to_llm_context() as grounding + question as user message
        # Output: natural language response string from LLM/SLM
        raise NotImplementedError("LLM integration not yet implemented.")
