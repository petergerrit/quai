"""Tests for clifford_orbit_distance: Frobenius distance from a candidate
unitary to the double-coset C · diag · C of a finite subgroup C.

Strategy: construct U = g · D · h for g, h in the registered base group and
D a rational-angle diagonal; the distance should collapse to ~1e-10. A
generic Haar draw should give a much larger residual (~0.5 in SU(3))."""
from __future__ import annotations

import math

import numpy as np
import pytest

from swiftbot.tools.groups import get_group, REGISTRY
from swiftbot.tools.orbit_distance import clifford_orbit_distance


D_DENOMS = (3, 4, 6, 8, 9, 12, 16, 18, 24)


@pytest.fixture(scope="module")
def C_S648():
    """Σ(216×3), the single-qutrit Clifford group (|C|=648)."""
    return list(get_group("S648"))


def test_structured_on_orbit_zero_residual(C_S648):
    """U = g · diag(1, ω_9, ω_9⁻¹) · h for some g, h ∈ C should hit
    residual ≈ 0 and flag as approximately distillable."""
    D = np.diag([1, np.exp(2j * math.pi / 9), np.exp(-2j * math.pi / 9)])
    g, h = C_S648[5], C_S648[123]
    U = g @ D @ h
    res = clifford_orbit_distance(U, C_S648, rational_denoms=D_DENOMS)
    assert res.residual < 1e-8, f"expected ≈0, got {res.residual}"
    assert res.approximately_distillable, "should be flagged as distillable"
    # All three phases should be snapped to a rational denominator
    assert all(d > 0 for d in res.best_rational_denoms)


def test_haar_random_large_residual(C_S648):
    """A Haar-random SU(3) extension should have large orbit distance."""
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    U = Q / np.linalg.det(Q) ** (1 / 3)

    res = clifford_orbit_distance(U, C_S648, rational_denoms=D_DENOMS)
    # Generic Haar on SU(3) gives residual ~0.3–0.8; definitely not distillable
    assert res.residual > 0.1, f"expected ≳0.1 for Haar, got {res.residual}"
    assert not res.approximately_distillable


def test_identity_is_trivially_distillable(C_S648):
    """U = I is in C (as C[0] if identity comes first, or via g · I · g†);
    residual must be 0 and all phases trivially rational (0 · π)."""
    U = np.eye(3, dtype=complex)
    res = clifford_orbit_distance(U, C_S648)
    assert res.residual < 1e-10
    # 0/π is trivially rational on any denominator
    for pp in res.D_phases_over_pi:
        assert abs(pp) < 1e-9


def test_shape_and_types(C_S648):
    U = np.eye(3, dtype=complex)
    res = clifford_orbit_distance(U, C_S648[:20])
    assert isinstance(res.residual, float)
    assert 0 <= res.g_index < 20
    assert 0 <= res.h_index < 20
    assert len(res.D_phases) == 3
    assert len(res.best_rational_denoms) == 3
    assert isinstance(res.approximately_distillable, bool)
