from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai


load_dotenv()


class LLMClient:
    """Small wrapper around the Google GenAI client."""

    def __init__(
        self,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.client = client or genai.Client()

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        """Generate text from a single prompt."""
        response = self.generate_raw(contents=prompt, model=model, **kwargs)
        return self._extract_text(response)

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
        return self._extract_text(response)

    def generate_with_system(
        self,
        prompt: str,
        system_instruction: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text with a system instruction."""
        config = dict(kwargs.pop("config", {}) or {})
        config["system_instruction"] = system_instruction

        response = self.generate_raw(
            contents=prompt,
            model=model,
            config=config,
            **kwargs,
        )
        return self._extract_text(response)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a multi-turn message list."""
        response = self.generate_raw(contents=messages, model=model, **kwargs)
        return self._extract_text(response)

    def generate_raw(
        self,
        contents: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the provider response without post-processing."""
        return self.client.models.generate_content(
            model=model or self.model,
            contents=contents,
            **kwargs,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
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


if __name__ == "__main__":
    llm = LLMClient()
    print(llm.generate("Explain how AI works in a few words."))
