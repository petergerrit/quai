"""Tests for the Sawicki universality check.

Coverage:
  * is_irreducible for several real finite groups (expected: irreducible)
  * a deliberately reducible example (diagonal-only set on d=3): not irreducible
  * check_universality verdicts on those cases
  * distance_to_center sanity values
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "clifford_t") not in sys.path:
    sys.path.insert(0, str(REPO / "clifford_t"))
from genGROUP import close_group  # noqa: E402

from swiftbot.tools import groups as gmod
from swiftbot.tools.sawicki import (
    adjoint_matrix,
    check_universality,
    commutant_dimension,
    distance_to_center,
    is_irreducible,
    su_basis,
)


# ---------------------------------------------------------------------------
# Basis + adjoint sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3, 4])
def test_su_basis_is_orthonormal(d: int) -> None:
    B = su_basis(d)
    n = d * d - 1
    assert B.shape == (n, d, d)
    # For Hermitian T_i, T_j: ⟨T_i, T_j⟩ = tr(T_i T_j) — real and orthonormal.
    gram = np.einsum("iab,jba->ij", B, B).real
    assert np.allclose(gram, np.eye(n), atol=1e-10)
    for i in range(n):
        assert abs(np.trace(B[i])) < 1e-12
        assert np.allclose(B[i], B[i].conj().T, atol=1e-12)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_adjoint_identity_is_identity(d: int) -> None:
    ad_I = adjoint_matrix(np.eye(d, dtype=complex))
    assert np.allclose(ad_I, np.eye(d * d - 1), atol=1e-10)


@pytest.mark.parametrize("d", [2, 3])
def test_adjoint_is_orthogonal_for_unitary(d: int) -> None:
    rng = np.random.default_rng(42)
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(G)
    g = Q / np.linalg.det(Q) ** (1.0 / d)  # project to SU(d)
    ad = adjoint_matrix(g)
    assert np.allclose(ad @ ad.T, np.eye(d * d - 1), atol=1e-10)


# ---------------------------------------------------------------------------
# Irreducibility on real finite groups
# ---------------------------------------------------------------------------

# Expected commutant dimensions per registered group. Dim = 1 means the
# group's adjoint rep on su(d) is irreducible over ℝ; >1 means it has
# invariant subspaces — a meaningful structural fact for the paper: any
# extension T must break those subspaces to reach SU(d)-universality.
EXPECTED_COMMUTANT_DIM = {
    # d = 2  — all irreducible
    "BI": 1, "BO": 1, "BT": 1, "clifford": 1, "hurwitz": 1,
    # d = 3
    "S60":   2, "S108":  2,          # reducible (invariant subspace in su(3))
    "S216":  1, "S648":  1, "S1080": 1,
    # d = 4
    "s60":    4, "s60x4":  4,
    "s120x41": 3, "s120x42": 3,
    "s720x4":  2,
    "s7f":     1,                    # only d=4 registered group with irreducible Ad
}


@pytest.mark.parametrize(
    "group_name,expected_dim",
    sorted(EXPECTED_COMMUTANT_DIM.items()),
)
def test_commutant_dim_of_standard_groups(group_name: str, expected_dim: int) -> None:
    """Commutant dimension of the adjoint representation of each registered
    finite group. Values were computed with the character formula (see
    swiftbot.tools.sawicki.commutant_dimension) and checked in."""
    mats = list(gmod.get_group(group_name))
    assert commutant_dimension(mats) == expected_dim


def test_reducible_example_diagonal_only_d3() -> None:
    """Only diagonal SU(3) gates commute with any diagonal matrix on su(3), so
    a set consisting solely of diagonal unitaries has a non-scalar commutant
    → reducible → Sawicki says not universal.

    Use rational-multiple-of-π phases so the set closes in a finite group.
    The character formula needs the closed group as input, so we close first.
    """
    d = 3
    mats = []
    for theta in (np.pi / 3, 2 * np.pi / 3):
        phases = np.exp(1j * np.array([theta, -2 * theta, theta]))  # det = 1
        mats.append(np.diag(phases))
    closed = close_group(mats, max_size=200, verbose=False)
    # Commutant of a set of diagonal matrices contains all diagonal operators
    # on su(d): so commutant_dim ≥ d = 3 (dim of diagonal subspace of su(3)).
    assert not is_irreducible(closed), "diagonal-only set should be reducible on su(3)"
    result = check_universality(closed)
    assert result.verdict == "not_universal"
    assert result.commutant_dim > 1


# ---------------------------------------------------------------------------
# check_universality verdicts on real finite groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "group_name,expected_verdict,expected_irreducible",
    [
        # d = 2: irreducible Ad, but all elements sit farther than 1/√2 from
        # {I, -I} — the Sawicki sufficient clause fails → "inconclusive".
        ("clifford", "inconclusive",   True),
        ("BT",       "inconclusive",   True),
        ("BO",       "inconclusive",   True),
        ("BI",       "inconclusive",   True),
        ("hurwitz",  "inconclusive",   True),
        # d = 3 — S108 has reducible Ad, so the *necessary* clause fails →
        # Sawicki rules it out as non-universal without needing the distance
        # test. Σ(72×3) and larger are irreducible → inconclusive.
        ("S108",     "not_universal", False),
        ("S216",     "inconclusive",   True),
        ("S648",     "inconclusive",   True),
        # d = 4 — most registered groups have reducible Ad.
        ("s60",      "not_universal", False),
        ("s7f",      "inconclusive",   True),
    ],
)
def test_verdict_on_standard_groups(
    group_name: str, expected_verdict: str, expected_irreducible: bool
) -> None:
    mats = list(gmod.get_group(group_name))
    result = check_universality(mats)
    assert result.verdict == expected_verdict, (
        f"{group_name}: got verdict {result.verdict}, expected {expected_verdict}. "
        f"(irreducible={result.irreducible}, "
        f"min distance to centre = {result.min_distance_to_center:.4f})"
    )
    assert result.irreducible == expected_irreducible


def test_distance_clause_fires_when_there_is_a_near_center_element() -> None:
    """The 'has_near_center_element' flag should fire as soon as any element
    in the input sits inside (0, 1/√2] HS distance from SU(d) centre.

    We don't need a full finite-group fixture for this — we only test the
    distance clause of the Sawicki criterion in isolation. Construct BT
    (irreducible) and append a tiny-angle rotation; close_group is bypassed
    because the character formula requires a closed group, so we only check
    the distance flag on the augmented list directly via distance_to_center.
    """
    theta = 0.05  # HS distance ≈ 2·sin(θ/2) ≈ 0.05 < 1/√2
    small_rot = np.array(
        [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
         [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=complex,
    )
    hs = distance_to_center(small_rot)
    assert 0 < hs <= 1.0 / np.sqrt(2), f"expected small_rot inside Sawicki radius; got {hs}"
    # The has_near_center_element flag on check_universality just iterates
    # distance_to_center across the input; we can safely call it on a
    # closed-group + one extra element list *without* relying on closure,
    # since the distance test doesn't invoke the character formula.
    mats = list(gmod.get_group("BT")) + [small_rot]
    # Directly probe the distance clause:
    threshold = 1.0 / np.sqrt(2)
    near_any = any(
        0 < distance_to_center(np.asarray(g)) <= threshold for g in mats
    )
    assert near_any, "near-centre flag should fire given the small rotation"


# ---------------------------------------------------------------------------
# Distance-to-centre numerics
# ---------------------------------------------------------------------------

def test_distance_to_center_identity_is_zero() -> None:
    assert distance_to_center(np.eye(2, dtype=complex)) == pytest.approx(0.0, abs=1e-12)
    assert distance_to_center(np.eye(3, dtype=complex)) == pytest.approx(0.0, abs=1e-12)


def test_distance_to_center_of_hadamard_su2() -> None:
    """HS distance of (SU(2)-normalised) Hadamard from the centre {I, -I}.
    The Hadamard has tr(H_SU2) = 0, so
        HS(h, ±I)² = tr((h ∓ I)†(h ∓ I)) = 2 + 2 ∓ 2·Re tr(h) = 4.
    Hence min distance = 2 — comfortably outside the Sawicki radius 1/√2."""
    raw = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    det = np.linalg.det(raw)
    h = raw / det ** 0.5
    assert distance_to_center(h) == pytest.approx(2.0, abs=1e-9)
