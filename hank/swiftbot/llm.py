"""Thin wrapper around the Anthropic SDK for structured LLM calls.

SWIFTbot's "LLM at stage boundaries" policy means each LLM invocation
returns a typed Pydantic model, never free-form text that the pipeline has
to parse. We force structured output via the tool-use mechanism: the model
is given a single tool whose input schema matches the Pydantic schema, and
is required to call it (`tool_choice={"type":"tool", "name": ...}`).

Model identifiers (from Claude 4.5 / 4.6 lineup):
    * claude-opus-4-6       — most capable; supervisor at key decision points
    * claude-sonnet-4-6     — balanced default for research sweeps
    * claude-haiku-4-5-20251001  — cheap specialist for lookups / summaries

API key: read from ANTHROPIC_API_KEY in the environment; pass `api_key=`
to override. No Fermilab GPT-SSO path yet (by design decision).
"""
from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar, runtime_checkable

import anthropic
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "claude-sonnet-4-6"

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMBackend(Protocol):
    """The subset of the LLM surface that SWIFTbot actually uses.

    Implementations: `AnthropicLLM` (live SDK), `ScriptedLLM` (for tests)."""

    def ask_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        tool_name: str = ...,
        max_tokens: int = ...,
    ) -> T: ...


class AnthropicLLM:
    """Production backend: calls Anthropic's messages endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key=... "
                "(we don't pull it from any other source by design)."
            )
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)

    def ask_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        tool_name: str = "submit",
        max_tokens: int = 2048,
    ) -> T:
        """Force Claude to populate `output_model` by calling a tool."""
        schema = output_model.model_json_schema()
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            tool_choice={"type": "tool", "name": tool_name},
            tools=[{
                "name": tool_name,
                "description": (output_model.__doc__ or "Submit the structured answer.").strip(),
                "input_schema": schema,
            }],
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return _validate_with_string_fallback(output_model, block.input)
        raise RuntimeError(
            f"LLM did not invoke tool {tool_name!r}; content={resp.content!r}"
        )


def _validate_with_string_fallback(output_model: type[T], raw: dict) -> T:
    """Pydantic-validate `raw`. If it fails, try to coerce any string fields
    that look like JSON into parsed JSON and retry.

    Works around a tool-use quirk where the model sometimes serialises list
    or dict fields as JSON-encoded strings instead of native structures.
    """
    try:
        return output_model.model_validate(raw)
    except ValidationError:
        if not isinstance(raw, dict):
            raise
        coerced = dict(raw)
        for key, val in list(coerced.items()):
            if isinstance(val, str):
                stripped = val.lstrip()
                if stripped.startswith("[") or stripped.startswith("{"):
                    try:
                        coerced[key] = json.loads(val)
                    except json.JSONDecodeError:
                        pass
        return output_model.model_validate(coerced)


class ScriptedLLM:
    """Test backend: returns pre-registered responses in FIFO order.

    Useful for deterministic unit tests of the supervisor without any real
    API call.
    """

    def __init__(self, responses: list[BaseModel]) -> None:
        self._queue: list[BaseModel] = list(responses)
        self.history: list[dict] = []    # what was asked, for assertions

    def ask_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        tool_name: str = "submit",
        max_tokens: int = 2048,
    ) -> T:
        self.history.append(
            {"system": system, "user": user, "output_model": output_model.__name__}
        )
        if not self._queue:
            raise RuntimeError(
                f"ScriptedLLM exhausted — no pre-registered response for "
                f"output_model={output_model.__name__}"
            )
        result = self._queue.pop(0)
        if not isinstance(result, output_model):
            raise TypeError(
                f"Next scripted response is {type(result).__name__}; "
                f"expected {output_model.__name__}"
            )
        return result
