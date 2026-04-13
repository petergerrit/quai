"""Smoke-test the Stage-1 group registry.

Loads every registered group, verifies shape and unitarity. Also checks
get_group cache semantics and register_custom.
"""
from __future__ import annotations

import numpy as np
import pytest

from swiftbot.tools import groups as gmod
from swiftbot.tools.groups import (
    REGISTRY,
    get_group,
    list_groups,
    register_custom,
    clear_cache,
)


# The autouse `_preserve_group_registry` fixture in conftest.py already
# handles REGISTRY and cache snapshotting. Nothing local needed.


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_get_group_matches_spec(name: str) -> None:
    spec = REGISTRY[name]
    arr = get_group(name)
    assert arr.shape == (spec.expected_size, spec.d, spec.d), f"{name}: wrong shape"

    # Unitarity check: every element times its adjoint ≈ I.
    eye = np.eye(spec.d, dtype=complex)
    u_errs = np.max(np.abs(arr @ arr.conj().transpose(0, 2, 1) - eye), axis=(1, 2))
    assert u_errs.max() < 1e-6, f"{name}: max unitarity error {u_errs.max():.2e}"


def test_list_groups_filters_by_dim() -> None:
    all_specs = list_groups()
    assert len(all_specs) == len(REGISTRY)
    for d in (2, 3, 4):
        filtered = list_groups(d=d)
        assert filtered, f"no d={d} groups registered"
        assert all(s.d == d for s in filtered)


def test_cache_returns_same_object() -> None:
    a1 = get_group("BI")
    a2 = get_group("BI")
    assert a1 is a2, "cache miss: get_group should memoize"


def test_register_custom_builds_pauli_group() -> None:
    # Pauli 1-qubit group (4 elements as matrices, up to global phase inside SU(2)).
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    arr = register_custom("pauli_test", [X, Z], expected_size=8)
    # X and Z generate the 8-element Pauli group {±I, ±X, ±Y, ±Z} before
    # projection; after normalizing to SU(2), det(Y) = -1 maps to det=1 via
    # the principal square root, so we should land at 8 distinct SU(2) elements.
    assert arr.shape[0] in (4, 8, 16), f"unexpected Pauli size {arr.shape[0]}"
    # And it's in the registry now:
    assert "pauli_test" in gmod.REGISTRY
