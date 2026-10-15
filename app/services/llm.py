"""Chat-completion client for any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, ...)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI, OpenAIError

from app.core.exceptions import ExternalServiceError

Message = dict[str, Any]
ToolSpec = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    async def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse: ...


class OpenAICompatibleLLM:
    def __init__(
        self, api_key: str, model: str, temperature: float, base_url: str | None = None
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    async def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise ExternalServiceError(f"LLM request failed: {exc}") from exc

        message = completion.choices[0].message
        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            fn = getattr(call, "function", None)
            if fn is None:
                continue
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=call.id, name=fn.name, arguments=args))
        return LLMResponse(content=message.content, tool_calls=calls)
