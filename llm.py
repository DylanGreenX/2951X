from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import config

load_dotenv()


class LLMClientError(RuntimeError):
    """Raised when the provider client cannot complete a model request."""


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
        """Return the provider response without post-processing."""
        try:
            return self.client.models.generate_content(
                model=model or self.model,
                contents=contents,
                **kwargs,
            )
        except Exception as exc:
            raise LLMClientError(f"Gemini generate_content failed: {exc}") from exc

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
        """Best-effort conversion of a GenAI response into plain text."""
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

        if parts:
            return "\n".join(parts)

        return str(response)

    _extract_text = extract_text


if __name__ == "__main__":
    llm = LLMClient()
    print(llm.generate("Explain how AI works in a few words."))
