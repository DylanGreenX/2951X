"""
Interaction system between player and NPC.

Handles proximity detection, the deterministic question/response cycle,
and the LLM/SLM path with structured message building and tool calling.

Response pipeline (LLM/SLM modes)
──────────────────────────────────
  get_llm_response()
    └─ _build_messages()          builds [system, user] + filtered tool list
         ├─ _npc_id_for()         resolves brain → registered npc_id
         ├─ _build_system_prompt() persona + knowledge constraints + observations
         └─ _get_tool_schemas()   filters get_all_objects for embodied mode
    └─ _call_llm() / _call_slm()
         ├─ _call_llm()           agentic tool-call loop (dispatch → re-query)
         │    └─ _invoke_model()  TODO(Joey): single LLM API call
         └─ _call_slm()           single-shot, no tool loop
              └─ _invoke_slm()    TODO(Gordan): single SLM API call
"""
from __future__ import annotations

import json
from typing import Any, NamedTuple, Optional

from entities import Player, NPC
from npc_brain import NPCBrain
from rlang_engine import get_natural_object_name, get_natural_location_name
from game_api_interface import GameAPIProvider, GAME_TOOL_SCHEMAS, dispatch_tool_call
import config


class ModelResponse(NamedTuple):
    """
    Normalised return value from a single model call.

    Decouples the tool-call loop from the exact response format of any
    particular LLM API. Joey's _invoke_model and Gordan's _invoke_slm
    both return this type.

    Fields
    ──────
    content    : The model's text reply, or None when it wants to call tools.
    tool_calls : List of {"id": str, "name": str, "arguments": str (raw JSON)}.
                 Empty list means the model returned a final text response.
    """
    content: Optional[str]
    tool_calls: list[dict[str, Any]]


class InteractionManager:
    def __init__(self, api: Optional[GameAPIProvider] = None) -> None:
        """
        Args:
            api: A concrete GameAPIProvider instance. Required for LLM/SLM
                 response modes; ignored for deterministic mode. Pass
                 PygameGameAPI.from_game(world, player, brain) here.
        """
        self.api = api

    # ── Public interface ──────────────────────────────────────────────────────

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

    # ── Deterministic baseline ────────────────────────────────────────────────

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

    # ── LLM / SLM entry point ─────────────────────────────────────────────────

    def get_llm_response(
        self, brain: NPCBrain, target_color: str, target_shape: str, model: str
    ) -> str:
        """
        Build structured messages from NPC knowledge and dispatch to the
        appropriate model path.

        The messages list and tool list are constructed once here and then
        passed into the model call, keeping prompt logic separate from
        HTTP/inference logic.
        """
        messages, tools = self._build_messages(brain, target_color, target_shape)

        if model == "llm":
            return self._call_llm(messages, tools)
        elif model == "slm":
            return self._call_slm(messages)
        raise ValueError(f"Unknown model: {model}")

    # ── Message / prompt construction ─────────────────────────────────────────

    def _npc_id_for(self, brain: NPCBrain) -> str:
        """
        Reverse-lookup the npc_id registered in self.api for a given brain
        reference. Falls back to "npc_0" (the PygameGameAPI default) when
        no api is set or the brain is not found — e.g. during unit tests
        that construct an InteractionManager without a live API.
        """
        if self.api is not None and hasattr(self.api, "brains"):
            for npc_id, registered_brain in self.api.brains.items():
                if registered_brain is brain:
                    return npc_id
        return "npc_0"

    def _get_tool_schemas(self, is_embodied: bool) -> list[dict[str, Any]]:
        """
        Return the tool list appropriate for the current knowledge mode.

        get_all_objects is excluded for embodied NPCs — leaving it in would
        give the model a backdoor to perfect knowledge, collapsing the
        embodied/perfect distinction that is the core independent variable
        of the experiment matrix.
        """
        if is_embodied:
            return [
                t for t in GAME_TOOL_SCHEMAS
                if t["function"]["name"] != "get_all_objects"
            ]
        return list(GAME_TOOL_SCHEMAS)

    def _build_system_prompt(
        self,
        npc_id: str,
        target_natural_name: str,
        context_lines: list[str],
        is_embodied: bool,
        is_competitive: bool,
    ) -> str:
        """
        Build the system message that establishes NPC persona, knowledge
        constraints, and experiment-condition-specific behaviour.

        Separating this from the user message is required for tool calling:
        the system message is injected once and persists across the entire
        tool-call loop; the user message carries only the player's question.

        Experiment conditions handled here
        ────────────────────────────────────
        is_embodied=True   : NPC must use get_npc_memory / get_exploration_status;
                             forbidden from using get_all_objects.
        is_embodied=False  : NPC may use get_all_objects for perfect knowledge.
        is_competitive=True: NPC is told to strategically withhold the target
                             location (CompetitiveNPCBrain condition).
        """
        context_str = "\n".join(context_lines)

        if is_embodied:
            knowledge_instruction = (
                "You can ONLY share information about items and locations you have "
                "personally witnessed during your travels. "
                "Never fabricate or guess at locations you have not visited. "
                "Use the get_npc_memory tool to consult your observations and "
                "get_exploration_status to acknowledge the limits of what you know."
            )
        else:
            knowledge_instruction = (
                "You have complete knowledge of this region. "
                "Use the get_all_objects tool to locate any item with certainty."
            )

        if is_competitive:
            sharing_instruction = (
                f"\nIMPORTANT: You are also searching for the {target_natural_name}. "
                "You are competing with this traveler. Give deliberately vague or "
                "misleading directions — you know where it is but will not reveal it."
            )
        else:
            sharing_instruction = ""

        return (
            f"You are a seasoned traveler in Skyrim. {knowledge_instruction}"
            f"{sharing_instruction}\n\n"
            f"Your current observations:\n{context_str}\n\n"
            f"Your NPC identifier is {npc_id!r}. Use it when calling NPC-specific tools.\n"
            "Respond in 1–2 sentences, in character."
        )

    def _build_messages(
        self,
        brain: NPCBrain,
        target_color: str,
        target_shape: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Build the (messages, tools) pair for a single player interaction.

          messages[0]  system — NPC persona, knowledge constraints, observations
          messages[1]  user   — the player's question

        tools is the filtered GAME_TOOL_SCHEMAS list for this experiment condition.

        Returns:
            (messages, tools) both ready to pass directly to the model API.
        """
        label = f"{target_color}_{target_shape}"
        natural_name = get_natural_object_name(label)
        npc_id = self._npc_id_for(brain)
        is_embodied = getattr(config, "NPC_KNOWLEDGE_MODE", "embodied") == "embodied"
        is_competitive = getattr(config, "NPC_COMPETING", False)

        # Respect CompetitiveNPCBrain's sharing filter for the context lines
        # that get injected into the system prompt (knowledge withheld from LLM).
        if hasattr(brain, "get_sharing_context"):
            context_lines = brain.get_sharing_context(target_color, target_shape)
        else:
            context_lines = brain.state.to_llm_context()

        system_content = self._build_system_prompt(
            npc_id=npc_id,
            target_natural_name=natural_name,
            context_lines=context_lines,
            is_embodied=is_embodied,
            is_competitive=is_competitive,
        )
        tools = self._get_tool_schemas(is_embodied)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": f"Where is the {natural_name}?"},
        ]
        return messages, tools

    # ── Model calls ───────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str:
        """
        Agentic tool-call loop for the full LLM.

        Repeatedly calls _invoke_model. If the model returns tool_calls,
        each call is executed via dispatch_tool_call and its result appended
        to the message thread. Loops until the model returns a plain text
        response with no tool calls.
        """
        if self.api is None:
            raise RuntimeError(
                "InteractionManager.api is None — pass a GameAPIProvider to "
                "InteractionManager.__init__ before using LLM mode."
            )

        while True:
            model_response = self._invoke_model(messages, tools)

            if not model_response.tool_calls:
                # Model produced a final text response.
                return model_response.content or ""

            # Append the assistant's tool-calling turn to the thread.
            messages.append({
                "role":    "assistant",
                "content": model_response.content,  # may be None
                "tool_calls": [
                    {
                        "id":       tc["id"],
                        "type":     "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in model_response.tool_calls
                ],
            })

            # Execute each tool call and append results for the next turn.
            for tc in model_response.tool_calls:
                try:
                    result = dispatch_tool_call(
                        self.api,
                        tc["name"],
                        json.loads(tc["arguments"]),
                    )
                except (ValueError, KeyError) as exc:
                    result = {"error": str(exc)}

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      json.dumps(result),
                })

    def _call_slm(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """
        Single-shot call for the SLM.

        Most small models don't support tool calling, so all context is
        injected up-front in messages[0] (the system prompt) rather than
        fetched on demand. No loop — one call, one response.
        """
        return self._invoke_slm(messages)

    # ── Model stubs (to be implemented by teammates) ──────────────────────────

    def _invoke_model(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """
        TODO(Joey): Make one API call to the full LLM and return a ModelResponse.

        Implementation guide
        ────────────────────
        import openai

        response = openai.chat.completions.create(
            model="gpt-4o",           # or whichever model you choose
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        return ModelResponse(
            content=msg.content,      # None when the model wants to call tools
            tool_calls=[
                {
                    "id":        tc.id,
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,  # raw JSON string
                }
                for tc in (msg.tool_calls or [])
            ],
        )

        Notes
        ─────
        - `messages` and `tools` are already fully constructed — don't modify them.
        - If the model returns tool_calls, _call_llm will handle dispatch and
          re-call _invoke_model automatically. You only need one API call here.
        - response time measurement should wrap this method's call site in
          experiment.py, not inside this method.
        """
        raise NotImplementedError("LLM integration not yet implemented. See TODO(Joey).")

    def _invoke_slm(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """
        TODO(Gordan): Make one call to the SLM and return the response string.

        Implementation guide
        ────────────────────
        messages[0]  system prompt — NPC persona + RLang observations
        messages[1]  user message  — "Where is the crimson flag?"

        Option A — OpenAI-compatible local server (Ollama, llama.cpp, LM Studio):

            import openai
            client = openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            response = client.chat.completions.create(
                model="tinyllama",    # or "phi3", "mistral", etc.
                messages=messages,
            )
            return response.choices[0].message.content

        Option B — HuggingFace pipeline:

            from transformers import pipeline
            # Build a single prompt string from messages using the model's
            # chat template, then run through the text-generation pipeline.
            pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
            result = pipe(prompt, max_new_tokens=128)
            return result[0]["generated_text"].split("assistant")[-1].strip()

        Notes
        ─────
        - Tool calling is intentionally not supported here. The full context
          is already in messages[0]. If tool support is added later, promote
          this to a _call_slm loop (matching _call_llm) and return ModelResponse.
        """
        raise NotImplementedError("SLM integration not yet implemented. See TODO(Gordan).")
