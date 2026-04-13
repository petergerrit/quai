"""Tests for ask_structured's tolerance of tool-use quirks.

Some Claude tool-use calls return complex fields as JSON-encoded strings
rather than native lists/dicts. ask_structured coerces them back.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from swiftbot.llm import _validate_with_string_fallback


class Item(BaseModel):
    name: str


class Wrapper(BaseModel):
    items: list[Item]


def test_native_dict_still_works() -> None:
    result = _validate_with_string_fallback(
        Wrapper, {"items": [{"name": "a"}, {"name": "b"}]}
    )
    assert [i.name for i in result.items] == ["a", "b"]


def test_string_encoded_list_is_coerced() -> None:
    """Simulates the Claude tool-use quirk: the list field arrives as a
    JSON-encoded string."""
    result = _validate_with_string_fallback(
        Wrapper, {"items": '[{"name": "a"}, {"name": "b"}]'}
    )
    assert [i.name for i in result.items] == ["a", "b"]


def test_string_encoded_list_with_whitespace() -> None:
    result = _validate_with_string_fallback(
        Wrapper, {"items": "   [{\"name\": \"x\"}]"}
    )
    assert result.items[0].name == "x"


def test_bad_json_in_string_field_re_raises_original_error() -> None:
    with pytest.raises(Exception):
        _validate_with_string_fallback(Wrapper, {"items": "not valid json"})


def test_non_dict_input_is_not_coerced() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _validate_with_string_fallback(Wrapper, "not a dict")
