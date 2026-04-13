"""Shared pytest fixtures.

The group registry is a module-level dict that `register_custom` mutates; without
a test-level snapshot/restore, one test's ad-hoc registration leaks into later
tests (e.g. `test_register_custom_builds_pauli_group` poisons `list_groups(d=2)`).
"""
from __future__ import annotations

import pytest

from swiftbot.tools import groups as gmod


@pytest.fixture(autouse=True)
def _preserve_group_registry():
    """Snapshot gmod.REGISTRY, the in-memory cache, and the inline-generator
    dispatch table around every test."""
    before_registry = dict(gmod.REGISTRY)
    before_inline = dict(gmod._INLINE_GEN_FNS)
    gmod.clear_cache()
    try:
        yield
    finally:
        gmod.REGISTRY.clear()
        gmod.REGISTRY.update(before_registry)
        gmod._INLINE_GEN_FNS.clear()
        gmod._INLINE_GEN_FNS.update(before_inline)
        gmod.clear_cache()
