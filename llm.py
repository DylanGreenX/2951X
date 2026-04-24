from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import config

load_dotenv()


# Free-tier Gemini returns 429 RESOURCE_EXHAUSTED when RPM or daily quota is
# hit. 429s during a short burst are almost always the per-minute limit, so
# a few retries with backoff usually clears them without human intervention.
# Daily-quota 429s still bubble up after the retry budget, preserving the
# fail-loud behaviour experiments need.
_MAX_RETRIES = 5
_RETRY_BASE_S = 4.0

_RETRYABLE_STATUS_RE = re.compile(r"\b(429|500|503)\b")


class LLMClientError(RuntimeError):
    """Raised when the provider client cannot complete a model request."""


class SLMClientError(RuntimeError):
    """Raised when the local Hugging Face client cannot complete generation."""


@dataclass
class SLMResult:
    """Normalized result from a local SLM generation call."""

    text: str
    usage_metadata: dict[str, Any]
    model_id: str
    device: str
    generation_config: dict[str, Any]


class LLMClient:
    """Small wrapper around the Google GenAI client."""

    def __init__(
        self,
        model: str | None = None,
        client: genai.Client | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", config.DEFAULT_GEMINI_MODEL)
        self.timeout_ms = timeout_ms or int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
        try:
            self.client = client or genai.Client()
        except Exception as exc:  # Provider setup errors vary by SDK version.
            raise LLMClientError(f"Could not initialize Gemini client: {exc}") from exc

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        """Generate text from a single prompt."""
        response = self.generate_raw(contents=prompt, model=model, **kwargs)
        return self.extract_text(response)

    def generate_with_context(
        self,
        prompt: str,
        context: list[str] | tuple[str, ...],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using extra context strings plus the user prompt."""
        contents = [*context, prompt]
        response = self.generate_raw(contents=contents, model=model, **kwargs)
        return self.extract_text(response)

    def generate_with_system(
        self,
        prompt: str,
        system_instruction: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text with a system instruction."""
        response = self.generate_content(
            contents=prompt,
            model=model,
            system_instruction=system_instruction,
            **kwargs,
        )
        return self.extract_text(response)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a multi-turn message list."""
        response = self.generate_raw(contents=messages, model=model, **kwargs)
        return self.extract_text(response)

    def generate_content(
        self,
        contents: Any,
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
        model: str | None = None,
        config: types.GenerateContentConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate content with a normalized Gemini config."""
        gemini_config = self._build_config(
            config=config,
            system_instruction=system_instruction,
            tools=tools,
            **kwargs,
        )
        return self.generate_raw(
            contents=contents,
            model=model,
            config=gemini_config,
        )

    def generate_raw(
        self,
        contents: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the provider response without post-processing.

        Retries transient errors (429/500/503) with exponential backoff plus
        jitter — the common failure mode in experiments is bursting past the
        free-tier RPM limit, which clears on its own within a minute.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.models.generate_content(
                    model=model or self.model,
                    contents=contents,
                    **kwargs,
                )
            except Exception as exc:
                if not _RETRYABLE_STATUS_RE.search(str(exc)) or attempt == _MAX_RETRIES - 1:
                    raise LLMClientError(f"Gemini generate_content failed: {exc}") from exc
                delay = _RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                last_exc = exc
        raise LLMClientError(f"Gemini generate_content failed: {last_exc}")

    def _build_config(
        self,
        config: types.GenerateContentConfig | dict[str, Any] | None = None,
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
        **overrides: Any,
    ) -> types.GenerateContentConfig:
        """Merge caller config with provider defaults used by this demo."""
        if isinstance(config, types.GenerateContentConfig):
            data = config.model_dump(exclude_none=True)
        else:
            data = dict(config or {})

        data.update({k: v for k, v in overrides.items() if v is not None})
        if system_instruction is not None:
            data["system_instruction"] = system_instruction
        if tools is not None:
            data["tools"] = tools
        data.setdefault(
            "http_options",
            types.HttpOptions(timeout=self.timeout_ms),
        )
        data.setdefault(
            "automatic_function_calling",
            types.AutomaticFunctionCallingConfig(disable=True),
        )
        return types.GenerateContentConfig(**data)

    @staticmethod
    def user_content(text: str) -> types.Content:
        """Build a Gemini user content message."""
        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    @staticmethod
    def function_response_content(
        function_call: types.FunctionCall,
        response: dict[str, Any],
    ) -> types.Content:
        """Build a Gemini function-response content message."""
        return types.Content(
            role="user",
            parts=[LLMClient.function_response_part(function_call, response)],
        )

    @staticmethod
    def function_response_part(
        function_call: types.FunctionCall,
        response: dict[str, Any],
    ) -> types.Part:
        """Build a Gemini function response while preserving call IDs if present."""
        function_response = types.FunctionResponse(
            id=getattr(function_call, "id", None),
            name=getattr(function_call, "name", None) or "",
            response=response,
        )
        return types.Part(function_response=function_response)

    @staticmethod
    def function_response_content_from_parts(parts: list[types.Part]) -> types.Content:
        """Build one Gemini user turn containing one or more function responses."""
        return types.Content(role="user", parts=parts)

    @staticmethod
    def to_gemini_tools(tool_schemas: list[dict[str, Any]]) -> list[types.Tool]:
        """Convert OpenAI-shaped function schemas into Gemini Tool declarations."""
        declarations: list[types.FunctionDeclaration] = []
        for schema in tool_schemas:
            if schema.get("type") != "function":
                continue
            function = schema.get("function", {})
            name = function.get("name")
            if not name:
                continue
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=function.get("description"),
                    parameters_json_schema=function.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                )
            )
        return [types.Tool(function_declarations=declarations)] if declarations else []

    @staticmethod
    def extract_function_calls(response: Any) -> list[types.FunctionCall]:
        """Return function calls from a Gemini response."""
        function_calls = getattr(response, "function_calls", None)
        if function_calls:
            return list(function_calls)

        candidates = getattr(response, "candidates", None) or []
        calls: list[types.FunctionCall] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    calls.append(function_call)
        return calls

    @staticmethod
    def extract_model_content(response: Any) -> types.Content | None:
        """Return the exact model content that must be preserved in tool loops."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        return getattr(candidates[0], "content", None)

    @staticmethod
    def extract_usage_metadata(response: Any) -> dict[str, Any]:
        """Return provider token usage metadata as a plain dict when available."""
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump(mode="json", exclude_none=True)
        if isinstance(usage, dict):
            return dict(usage)
        return {
            name: getattr(usage, name)
            for name in dir(usage)
            if not name.startswith("_") and isinstance(getattr(usage, name), (int, float, str, bool))
        }

    @staticmethod
    def extract_text(response: Any) -> str:
        """Best-effort conversion of a GenAI response into plain text.

        Returns "" (not repr(response)) when the response carries no text
        parts — models that return empty content, burn their output budget
        on thinking tokens, or hit a safety filter should surface as an
        empty string, not a raw SDK debug dump leaking into experiments.
        """
        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        parts: list[str] = []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)

        return "\n".join(parts) if parts else ""

    _extract_text = extract_text


class SLMClient:
    """Lazy local Hugging Face causal-LM wrapper for SLM response modes."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv("NPC_SLM_MODEL_ID", "HuggingFaceTB/SmolLM-135M")
        self.device_preference = device
        self.dtype_preference = dtype
        self.tokenizer = tokenizer
        self.model = model
        self.device: str | None = None
        self._torch: Any | None = None

    def preload(self) -> None:
        """Download and load local model assets ahead of the first generation."""
        self._ensure_loaded()

    def generate(
        self,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
        *,
        use_chat_template: bool = False,
    ) -> SLMResult:
        """Generate a completion and return text plus local token accounting.

        When ``use_chat_template`` is True and the tokenizer exposes a chat
        template (instruct/chat checkpoints do; base models do not), the raw
        ``prompt`` is wrapped as a single-turn user message and rendered through
        the model's native turn tokens via ``apply_chat_template``. This is
        load-bearing for SmolLM2-Instruct: without it the instruct tuning never
        activates and the model falls back to base-completion behaviour.
        """
        self._ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None
        assert self._torch is not None
        assert self.device is not None

        generation_kwargs = dict(generation_config or {})
        max_new_tokens = int(generation_kwargs.pop("max_new_tokens", 96))
        do_sample = bool(generation_kwargs.pop("do_sample", False))
        temperature = float(generation_kwargs.pop("temperature", 0.2))
        top_p = float(generation_kwargs.pop("top_p", 0.9))

        model_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            model_kwargs["temperature"] = temperature
            model_kwargs["top_p"] = top_p

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            model_kwargs["pad_token_id"] = eos_token_id

        model_kwargs.update(generation_kwargs)

        try:
            if use_chat_template and getattr(self.tokenizer, "chat_template", None):
                # Render to string first, then tokenize via the normal path.
                # This gives us a BatchEncoding with attention_mask for free,
                # and avoids the tensor/dict return-shape differences between
                # transformers versions.
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = self.tokenizer(rendered, return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = self._move_inputs_to_device(inputs)
            input_ids = inputs["input_ids"]
            prompt_token_count = int(input_ids.shape[-1])

            with self._torch.inference_mode():
                outputs = self.model.generate(**inputs, **model_kwargs)

            output_ids = outputs[0]
            generated_ids = output_ids[prompt_token_count:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            raise SLMClientError(f"Local SLM generation failed: {exc}") from exc

        candidates_token_count = int(generated_ids.shape[-1])
        usage_metadata = {
            "prompt_token_count": prompt_token_count,
            "candidates_token_count": candidates_token_count,
            "total_token_count": prompt_token_count + candidates_token_count,
        }
        normalized_config = {
            **model_kwargs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        return SLMResult(
            text=text,
            usage_metadata=usage_metadata,
            model_id=self.model_id,
            device=self.device,
            generation_config=normalized_config,
        )

    def _ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            if self._torch is None:
                try:
                    import torch
                except Exception as exc:
                    raise SLMClientError(f"Could not import torch: {exc}") from exc
                self._torch = torch
            if self.device is None:
                self.device = self._resolve_device(self._torch)
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise SLMClientError(
                "Could not import local SLM dependencies. Install transformers, "
                "torch, and safetensors."
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(torch)
        model_kwargs = self._model_load_kwargs(torch)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise SLMClientError(f"Could not load local SLM '{self.model_id}': {exc}") from exc

    def _resolve_device(self, torch: Any) -> str:
        requested = (self.device_preference or "auto").lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _model_load_kwargs(self, torch: Any) -> dict[str, Any]:
        requested = (self.dtype_preference or "auto").lower()
        if requested == "auto":
            return {}

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if requested not in dtype_map:
            raise SLMClientError(
                f"Unsupported NPC_SLM_DTYPE '{self.dtype_preference}'. "
                f"Use one of: {sorted(dtype_map)} or 'auto'."
            )
        return {"torch_dtype": dtype_map[requested]}

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }


if __name__ == "__main__":
    llm = LLMClient()
    print(llm.generate("Explain how AI works in a few words."))
