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
         └─ _get_tool_schemas()   filters full-world tools for embodied mode
    └─ _call_llm() / _call_slm()
         ├─ _call_llm()           Gemini tool-call loop (dispatch → re-query)
         └─ _call_slm()           single-shot, no tool loop
              └─ _invoke_slm()    TODO(Gordan): single SLM API call
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from entities import Player, NPC
from npc_brain import NPCBrain
from rlang_engine import (
    extract_coordinates_from_text,
    extract_regions_from_text,
    get_natural_object_name,
    region_of,
)
from game_api_interface import (
    GameAPIProvider,
    dispatch_tool_call,
    get_natural_position_name,
    get_tool_schemas_for_knowledge_mode,
)
from llm import LLMClient, LLMClientError
import config


def reset_llm_log() -> None:
    """Overwrite the LLM interaction log at game startup."""
    if not getattr(config, "NPC_LLM_LOG_ENABLED", False):
        return
    path = Path(getattr(config, "NPC_LLM_LOG_PATH", "llm_interactions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class InteractionManager:
    def __init__(
        self,
        api: Optional[GameAPIProvider] = None,
        llm_client: Optional[LLMClient] = None,
        enforce_grounding: Optional[bool] = None,
    ) -> None:
        """
        Args:
            api: A concrete GameAPIProvider instance. Required for LLM/SLM
                 response modes; ignored for deterministic mode. Pass
                 PygameGameAPI.from_game(world, player, brain) here.
            llm_client: Optional injected Gemini client for tests.
            enforce_grounding: When True, replace embodied LLM replies that
                 mention unobserved coordinates/locations. Defaults to
                 config.PLAY_MODE so experiments can opt out and log raw output.
        """
        self.api = api
        self.llm_client = llm_client
        self.enforce_grounding = (
            getattr(config, "NPC_ENFORCE_GROUNDING", config.PLAY_MODE)
            if enforce_grounding is None
            else enforce_grounding
        )
        self.last_raw_response: str = ""
        self.last_response: str = ""
        self.last_tool_calls: list[dict[str, Any]] = []
        self.last_grounding_violation: bool = False
        self.last_grounding_violations: list[tuple[int, int]] = []
        self.last_llm_error: str | None = None
        self.last_token_usage: list[dict[str, Any]] = []
        self.last_token_usage_total: dict[str, int] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def can_interact(self, player: Player, npc: NPC) -> bool:
        """Player must be on the same cell as the NPC to interact."""
        return player.x == npc.x and player.y == npc.y

    def start_interaction(self, brain, target_color, target_shape):
        self._reset_trace()
        label = f"{target_color}_{target_shape}"
        natural_name = get_natural_object_name(label)
        question = f"Please show me where the {natural_name} is."

        if config.NPC_RESPONSE_MODE == "deterministic":
            response = self.get_deterministic_response(brain, target_color, target_shape)
            self.last_raw_response = response
            self.last_response = response
        elif config.NPC_RESPONSE_MODE in ("llm", "slm"):
            response = self.get_llm_response(brain, target_color, target_shape, model=config.NPC_RESPONSE_MODE)
        else:
            raise ValueError(f"Unknown NPC_RESPONSE_MODE: {config.NPC_RESPONSE_MODE}")
        return question, response

    def _reset_trace(self) -> None:
        self.last_raw_response = ""
        self.last_response = ""
        self.last_tool_calls = []
        self.last_grounding_violation = False
        self.last_grounding_violations = []
        self.last_llm_error = None
        self.last_token_usage = []
        self.last_token_usage_total = {}

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

        world_size = brain.state.world_size
        natural_locations = [
            get_natural_position_name(x, y, world_size) for x, y in locations
        ]
        count = len(locations)
        if count == 1:
            return f"Aye, I found a {natural_name} in {natural_locations[0]}."

        location_list = ", ".join(natural_locations)
        return f"I've seen {count} {natural_name}s in: {location_list}."

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
            target_label = f"{target_color}_{target_shape}"
            raw_response = self._call_llm(messages, tools, target_label)
            self.last_raw_response = raw_response
            response = self._apply_grounding_guard(brain, raw_response)
            self.last_response = response
            self._log_llm_event(
                "interaction_final",
                {
                    "raw_response": raw_response,
                    "final_response": response,
                    "grounding_violation": self.last_grounding_violation,
                    "grounding_violations": self.last_grounding_violations,
                    "tool_calls": self.last_tool_calls,
                    "token_usage": self.last_token_usage,
                    "token_usage_total": self.last_token_usage_total,
                    "llm_error": self.last_llm_error,
                },
            )
            return response
        elif model == "slm":
            raw_response = self._call_slm(messages)
            self.last_raw_response = raw_response
            response = self._apply_grounding_guard(brain, raw_response)
            self.last_response = response
            self._log_llm_event(
                "interaction_final",
                {
                    "raw_response": raw_response,
                    "final_response": response,
                    "grounding_violation": self.last_grounding_violation,
                    "grounding_violations": self.last_grounding_violations,
                    "tool_calls": self.last_tool_calls,
                    "token_usage": self.last_token_usage,
                    "token_usage_total": self.last_token_usage_total,
                    "llm_error": self.last_llm_error,
                },
            )
            return response
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

        Embodied NPCs cannot receive arbitrary map/object lookup tools, because
        those would be a backdoor to hidden world state.
        """
        return get_tool_schemas_for_knowledge_mode(is_embodied)

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

        The LLM only ever sees natural-language identifiers: the item is
        "crimson flag", locations are region phrases like "the far southeast
        corner". Internal labels (red_triangle) and raw coordinates never
        appear on this surface — grounding/scoring happens server-side.

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
                "get_exploration_status to acknowledge the limits of what you know. "
                "When you know where the item is, use set_npc_target to walk there "
                "rather than just describing the spot."
            )
        else:
            knowledge_instruction = (
                "You have complete knowledge of this region. "
                "Use the get_all_objects tool to locate any item with certainty, "
                "then set_npc_target to walk there."
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
            f"The traveler is asking about the {target_natural_name}.\n"
            f"Your NPC identifier is {npc_id!r}. Use it when calling NPC-specific tools.\n"
            "Respond in 1–2 sentences, in character. "
            "If the traveler asks you to show where something is, call set_npc_target "
            "to navigate there. Do not ask for confirmation or follow-up."
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
            {"role": "user",   "content": f"Please show me where the {natural_name} is."},
        ]
        return messages, tools

    # ── Model calls ───────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        target_label: str,
    ) -> str:
        """
        Gemini tool-call loop for the full LLM.

        The conversation history uses Gemini Content objects. When the model
        asks for function calls, this loop appends the exact model content from
        the response, dispatches each call against the live GameAPIProvider,
        then appends Gemini function-response parts for the next model turn.
        """
        if self.api is None:
            raise RuntimeError(
                "InteractionManager.api is None — pass a GameAPIProvider to "
                "InteractionManager.__init__ before using LLM mode."
            )

        llm_client = self._get_llm_client()
        system_instruction = messages[0]["content"]
        user_prompt = messages[1]["content"]
        contents = [llm_client.user_content(user_prompt)]
        gemini_tools = llm_client.to_gemini_tools(tools)
        generation_config = {
            "temperature": getattr(config, "NPC_LLM_TEMPERATURE", 0.4),
            "max_output_tokens": getattr(config, "NPC_LLM_MAX_OUTPUT_TOKENS", 128),
        }
        max_tool_turns = getattr(config, "NPC_LLM_MAX_TOOL_TURNS", 4)
        self._log_llm_event(
            "interaction_start",
            {
                "model": llm_client.model,
                "knowledge_mode": getattr(config, "NPC_KNOWLEDGE_MODE", "embodied"),
                "response_mode": getattr(config, "NPC_RESPONSE_MODE", "deterministic"),
                "target_label": target_label,
                "messages": messages,
                "tool_names": [tool["function"]["name"] for tool in tools],
                "generation_config": generation_config,
                "max_tool_turns": max_tool_turns,
            },
        )

        for turn in range(max_tool_turns + 1):
            self._log_llm_event(
                "model_request",
                {
                    "turn": turn,
                    "system_instruction": system_instruction,
                    "contents": contents,
                    "tool_names": [tool["function"]["name"] for tool in tools],
                    "generation_config": generation_config,
                },
            )
            try:
                response = llm_client.generate_content(
                    contents=contents,
                    system_instruction=system_instruction,
                    tools=gemini_tools,
                    config=generation_config,
                )
            except LLMClientError as exc:
                self.last_llm_error = str(exc)
                self._log_llm_event(
                    "model_error",
                    {"turn": turn, "error": self.last_llm_error},
                )
                return "I cannot gather my thoughts clearly right now."

            usage_metadata = llm_client.extract_usage_metadata(response)
            self._record_token_usage(usage_metadata)
            function_calls = llm_client.extract_function_calls(response)
            response_text = "" if function_calls else llm_client.extract_text(response).strip()
            self._log_llm_event(
                "model_response",
                {
                    "turn": turn,
                    "text": response_text,
                    "function_calls": function_calls,
                    "usage_metadata": usage_metadata,
                    "token_usage_total": self.last_token_usage_total,
                    "raw_response": response,
                },
            )
            if not function_calls:
                return response_text

            model_content = llm_client.extract_model_content(response)
            if model_content is not None:
                contents.append(model_content)

            if turn == max_tool_turns:
                self.last_llm_error = "Gemini exceeded max tool-call turns."
                self._log_llm_event(
                    "tool_turn_limit",
                    {"turn": turn, "error": self.last_llm_error},
                )
                return "I need more time to sort through what I have seen."

            function_response_parts = []
            for function_call in function_calls:
                name = getattr(function_call, "name", None) or ""
                args = getattr(function_call, "args", None) or {}
                if not isinstance(args, dict):
                    self.last_llm_error = f"Invalid arguments for {name}: {args!r}"
                    self._log_llm_event(
                        "tool_error",
                        {"turn": turn, "name": name, "arguments": args, "error": self.last_llm_error},
                    )
                    return "I cannot make sense of that recollection right now."

                trace_entry = {
                    "turn": turn,
                    "name": name,
                    "arguments": dict(args),
                }
                try:
                    result = dispatch_tool_call(self.api, name, dict(args))
                except (ValueError, KeyError, TypeError) as exc:
                    self.last_llm_error = str(exc)
                    trace_entry["error"] = str(exc)
                    self.last_tool_calls.append(trace_entry)
                    self._log_llm_event("tool_error", trace_entry)
                    return "I cannot make sense of that recollection right now."

                result = self._apply_sharing_policy_to_tool_result(
                    name,
                    result,
                    target_label,
                )
                trace_entry["result"] = result
                self.last_tool_calls.append(trace_entry)
                self._log_llm_event("tool_call", trace_entry)
                function_response_parts.append(
                    llm_client.function_response_part(function_call, result)
                )

            if function_response_parts:
                contents.append(
                    llm_client.function_response_content_from_parts(
                        function_response_parts
                    )
                )

        self.last_llm_error = "Gemini tool-call loop exited unexpectedly."
        self._log_llm_event("model_error", {"error": self.last_llm_error})
        return "I need more time to sort through what I have seen."

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = LLMClient(
                timeout_ms=getattr(config, "NPC_LLM_TIMEOUT_MS", 30000)
            )
        return self.llm_client

    def _apply_grounding_guard(self, brain: NPCBrain, response: str) -> str:
        """
        Prevent embodied-mode replies from naming unobserved map locations.

        Experiments can disable enforcement to keep raw hallucinations in the
        data, while still recording violation metadata on this manager.
        """
        is_embodied = getattr(config, "NPC_KNOWLEDGE_MODE", "embodied") == "embodied"
        if not response or not is_embodied:
            return response

        mentioned_locations = set(extract_coordinates_from_text(response))
        observed_locations = set(brain.state.observed_cells)
        violations = sorted(mentioned_locations - observed_locations)
        self.last_grounding_violations = violations
        self.last_grounding_violation = bool(violations)

        if violations and self.enforce_grounding:
            return (
                "I cannot place it with certainty. "
                f"I have only explored {brain.state.coverage:.0%} of this region, "
                "and I will not speak beyond what I have seen."
            )
        return response

    def _apply_sharing_policy_to_tool_result(
        self,
        tool_name: str,
        result: dict[str, Any],
        target_label: str,
    ) -> dict[str, Any]:
        """Prevent competitive NPCs from leaking the target through tool data.

        All matching is done on Skyrim natural names since that is the only
        identifier the LLM-facing tool payloads carry.
        """
        if not getattr(config, "NPC_COMPETING", False):
            return result

        redacted = dict(result)
        target_name = get_natural_object_name(target_label)
        target_name_lower = target_name.lower()

        if tool_name == "get_npc_memory":
            observations = {
                name: locs
                for name, locs in redacted.get("observations", {}).items()
                if name.lower() != target_name_lower
            }
            if len(observations) != len(redacted.get("observations", {})):
                redacted["observations"] = observations
                redacted["withheld_for_competition"] = target_name
            redacted["context_lines"] = [
                line
                for line in redacted.get("context_lines", [])
                if target_name_lower not in line.lower()
            ]
        elif tool_name in {"get_all_objects", "get_nearby_objects"}:
            objects = [
                obj
                for obj in redacted.get("objects", [])
                if obj.get("name", "").lower() != target_name_lower
            ]
            if len(objects) != len(redacted.get("objects", [])):
                redacted["objects"] = objects
                redacted["withheld_for_competition"] = target_name
        elif tool_name == "get_object_at":
            obj = redacted.get("object")
            if obj and obj.get("name", "").lower() == target_name_lower:
                redacted["object"] = None
                redacted["withheld_for_competition"] = target_name

        return redacted

    def _record_token_usage(self, usage_metadata: dict[str, Any]) -> None:
        """Track per-turn and cumulative token usage from Gemini metadata."""
        if not usage_metadata:
            return

        self.last_token_usage.append(usage_metadata)
        for source_key, total_key in {
            "prompt_token_count": "prompt_token_count",
            "candidates_token_count": "candidates_token_count",
            "total_token_count": "total_token_count",
            "cached_content_token_count": "cached_content_token_count",
            "promptTokenCount": "prompt_token_count",
            "candidatesTokenCount": "candidates_token_count",
            "totalTokenCount": "total_token_count",
            "cachedContentTokenCount": "cached_content_token_count",
        }.items():
            value = usage_metadata.get(source_key)
            if isinstance(value, int):
                self.last_token_usage_total[total_key] = (
                    self.last_token_usage_total.get(total_key, 0) + value
                )

    def _log_llm_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one structured LLM event to the configured JSONL log."""
        if not getattr(config, "NPC_LLM_LOG_ENABLED", False):
            return

        path = Path(getattr(config, "NPC_LLM_LOG_PATH", "llm_interactions.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._json_safe(record), ensure_ascii=True) + "\n")

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert SDK objects into JSON-safe values for logs."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return cls._json_safe(value.model_dump(mode="json", exclude_none=True))
        return repr(value)

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

    # ── SLM stub (deferred) ───────────────────────────────────────────────────

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
          this to a _call_slm loop matching _call_llm.
        """
        raise NotImplementedError("SLM integration not yet implemented. See TODO(Gordan).")
