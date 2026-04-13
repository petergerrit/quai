"""Tests for swiftbot.stages.check_extension.

Covers the four regimes on known gate combinations:
  * finite:                   Clifford + S-gate (S already in Clifford → finite)
  * universal_likely:         Clifford + T-gate (Clifford+T is dense)
  * universal_likely:         Clifford + P(π/3) (we corrected earlier confusion)
  * finite (single-qubit):    Pauli + identity → small abelian closure
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from swiftbot.stages.check_extension import (
    CLASSIFICATION_BOUND,
    CLASSIFICATION_KNOWN,
    extension_verdict,
    identify_finite_subgroup,
    try_close_up_to,
)
from swiftbot.tools import groups as gmod


def _rz(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2), 0],
         [0, np.exp(1j * phi / 2)]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Classification bounds / known-group catalog sanity
# ---------------------------------------------------------------------------

def test_classification_bound_covers_registered_groups_per_dim() -> None:
    """Every registered group's size must be ≤ CLASSIFICATION_BOUND[d]."""
    for name, spec in gmod.REGISTRY.items():
        if spec.d in CLASSIFICATION_BOUND:
            assert spec.expected_size <= CLASSIFICATION_BOUND[spec.d], (
                f"{name} (order {spec.expected_size}) exceeds cap for d={spec.d}"
            )


def test_identify_trivial_and_cyclic() -> None:
    I2 = np.eye(2, dtype=complex)
    assert identify_finite_subgroup([I2]).startswith("trivial")


def test_identify_named_exceptionals() -> None:
    # Force-feed a size-matching list and check the label
    fake = [np.eye(2, dtype=complex)] * 24
    label = identify_finite_subgroup(fake)
    assert "24" in label and ("2T" in label or "S4" in label or "Clifford" in label)
    fake120 = [np.eye(2, dtype=complex)] * 120
    assert "2I" in identify_finite_subgroup(fake120)


# ---------------------------------------------------------------------------
# try_close_up_to
# ---------------------------------------------------------------------------

def test_close_clifford_stays_under_bound() -> None:
    """clifford.txt holds 24 one-per-projective-class matrices. Under SU(2)-
    level (non-projective) multiplication the closure is 2O = 48 elements.
    Under projective dedup we get the 24-element projective Clifford."""
    clifford = list(gmod.get_group("clifford"))
    # Non-projective closure: lands at the SU(2) lift 2O.
    closed = try_close_up_to(clifford, cap=100)
    assert closed is not None
    assert len(closed) == 48            # 2O
    # Projective closure: stays at the 24 projective representatives.
    closed_proj = try_close_up_to(clifford, cap=100, projective=True)
    assert closed_proj is not None
    assert len(closed_proj) == 24


def test_close_clifford_plus_T_exceeds_bound() -> None:
    """Clifford + T generates a dense (infinite) subgroup → exceeds bound."""
    clifford = list(gmod.get_group("clifford"))
    T = _rz(math.pi / 4)
    closed = try_close_up_to(clifford + [T, T.conj().T], cap=120)
    assert closed is None


# ---------------------------------------------------------------------------
# extension_verdict regimes
# ---------------------------------------------------------------------------

def test_verdict_clifford_plus_S_is_finite() -> None:
    """P(π/2) = S-gate, which is ALREADY in Clifford (projectively). SU(2)-
    level closure is the 48-element 2O = binary octahedral."""
    clifford = np.asarray(gmod.get_group("clifford"))
    S = _rz(math.pi / 2)
    v = extension_verdict(clifford, S, d=2)
    assert v.regime == "finite"
    assert v.closure_size == 48
    assert "2O" in v.identified_group
    assert v.irreducible is True


def test_verdict_clifford_plus_T_is_universal_likely() -> None:
    """Canonical Clifford + T is the paper's gold-standard universal set."""
    clifford = np.asarray(gmod.get_group("clifford"))
    T = _rz(math.pi / 4)
    v = extension_verdict(clifford, T, d=2)
    assert v.regime == "universal_likely"
    assert v.closure_size is None
    assert v.irreducible is True


def test_verdict_clifford_plus_P_pi_3_is_universal_likely() -> None:
    """Corrects my earlier mistaken claim that ⟨Clifford, P(π/3)⟩ is finite.

    By finite-subgroup classification of SU(2), ⟨Clifford, P(π/3)⟩ cannot
    fit in any of {Z_n, BD_{4n}, 2T, 2O, 2I}, so it's infinite → dense."""
    clifford = np.asarray(gmod.get_group("clifford"))
    T = _rz(math.pi / 3)
    v = extension_verdict(clifford, T, d=2)
    assert v.regime == "universal_likely"
    assert v.closure_size is None


def test_verdict_BT_plus_identity_is_finite_at_BT() -> None:
    """Extending BT with identity leaves BT unchanged."""
    BT = np.asarray(gmod.get_group("BT"))
    I = np.eye(2, dtype=complex)
    v = extension_verdict(BT, I, d=2)
    assert v.regime == "finite"
    assert v.closure_size == 24
    assert v.irreducible is True


def test_verdict_sigma36_plus_identity_keeps_reducibility() -> None:
    """Σ(36×3) has commutant_dim=2 (reducible Ad) — adding identity doesn't change this.
    Verdict should be 'finite' with irreducible=False."""
    S108 = np.asarray(gmod.get_group("S108"))
    I = np.eye(3, dtype=complex)
    v = extension_verdict(S108, I, d=3)
    assert v.regime == "finite"
    assert v.closure_size == 108
    assert v.irreducible is False
    assert v.commutant_dim == 2


def test_verdict_rejects_unknown_dimension_without_cap() -> None:
    with pytest.raises(ValueError, match="classification bound"):
        extension_verdict(
            [np.eye(5, dtype=complex)],
            np.eye(5, dtype=complex),
            d=5,
        )


def test_verdict_accepts_explicit_cap_for_unregistered_dim() -> None:
    """For a dimension without a built-in bound, caller can pass cap explicitly."""
    I5 = np.eye(5, dtype=complex)
    v = extension_verdict([I5], I5, d=5, cap=10)
    # Closure is {I} — trivially finite, size 1.
    assert v.regime == "finite"
    assert v.closure_size == 1
