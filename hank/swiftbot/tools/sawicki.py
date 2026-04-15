"""Sawicki universality check for finite matrix sets.

Implements the two-part criterion from
    A. Sawicki and K. Karnas,
    "Criteria for universality of quantum gates",
    Phys. Rev. A 95, 062303 (2017). arXiv:1610.00547

Criterion (for G = SU(d) acting by conjugation on su(d)):
    (A) the commutant of {Ad_g : g ∈ S} in End(su(d)) is 1-dimensional
    (B) some g ∈ S has Hilbert-Schmidt distance from the centre of SU(d)
        that lies in (0, 1/√2]

(A) alone is NECESSARY for universality. (A) AND (B) together are SUFFICIENT.
Consequences for the pipeline:
    - Irreducibility fails (A violated) → S is definitely not universal.
    - Both (A) and (B) hold → S is definitely universal.
    - (A) holds, (B) fails → inconclusive. Many perfectly universal gate sets
      (e.g. Clifford+T) fall here because their elements sit further than
      1/√2 from the SU(2) centre. In SWIFTbot we classify these by trying
      close_group with a large max_size: if closure terminates, the group is
      finite (and hence not universal); if it doesn't, we classify as
      "likely universal" and hand off to the Q_T branch.

Shape contract:
    Input `matrices` is a sequence of d×d unitaries (np.ndarray, complex).
    No identity-or-phase prefiltering is required; the algorithm handles it.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# su(d) basis + adjoint representation
# ---------------------------------------------------------------------------

def su_basis(d: int) -> np.ndarray:
    """Return (d²-1, d, d) array of traceless Hermitian matrices forming an
    orthonormal basis of su(d) under ⟨A, B⟩ := Re tr(A B†)."""
    if d < 2:
        raise ValueError(f"d must be ≥ 2; got {d}")
    basis: list[np.ndarray] = []
    # Symmetric off-diagonal (generalized Pauli X-like)
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = 1.0 / np.sqrt(2)
            m[k, j] = 1.0 / np.sqrt(2)
            basis.append(m)
    # Antisymmetric off-diagonal (Pauli Y-like)
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = -1j / np.sqrt(2)
            m[k, j] = 1j / np.sqrt(2)
            basis.append(m)
    # Diagonal (Pauli Z / generalized Gell-Mann diagonals)
    for l in range(1, d):
        m = np.zeros((d, d), dtype=complex)
        for i in range(l):
            m[i, i] = 1.0
        m[l, l] = -float(l)
        m = m / np.sqrt(l * (l + 1))
        basis.append(m)
    arr = np.asarray(basis, dtype=complex)
    assert arr.shape == (d * d - 1, d, d)
    return arr


def adjoint_matrix(g: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Matrix of Ad_g (conjugation by g) in the given orthonormal basis of su(d).

    For Hermitian basis elements T_k, the inner product reduces to
        ⟨T_i, X⟩ = tr(T_i X)    (real when X is Hermitian).
    So (Ad_g)_{ij} = tr(T_i · g T_j g†) — no complex conjugate on the basis.
    Ad_g ∈ SO(d²-1) for g ∈ SU(d); the matrix is strictly real."""
    g = np.asarray(g, dtype=complex)
    d = g.shape[0]
    if basis is None:
        basis = su_basis(d)
    g_inv = g.conj().T
    conjugated = np.einsum("ij,kjl,lm->kim", g, basis, g_inv)    # g T_k g†
    return np.einsum("iab,jba->ij", basis, conjugated).real      # tr(T_i · ...)


# ---------------------------------------------------------------------------
# Commutant / irreducibility
# ---------------------------------------------------------------------------

def commutant_dimension(
    matrices: Sequence[np.ndarray],
    *,
    integer_tol: float = 1e-6,
) -> int:
    """Dimension of the common commutant of {Ad_g : g ∈ matrices} on su(d).

    Uses the Schur / character formula for a **closed finite group** G:
        dim(commutant) = (1/|G|) · Σ_{g∈G} |χ(g)|²     with χ(g) = tr(Ad_g).

    This avoids the O(|G|·(d²-1)⁴) memory explosion of the naïve
    null-space-of-stacked-Kronecker approach; here memory is O(d²) and time
    is O(|G|·d⁶) (one (d²-1)×(d²-1) adjoint matrix held at a time).

    **Precondition**: `matrices` must be the **full closed finite group**
    ⟨S⟩ itself, not just generators — the character formula only holds under
    closure. Use clifford_t/genGROUP.close_group or groups.get_group to
    guarantee that.

    A non-integer result (beyond `integer_tol`) signals that the input is
    not a closed group (or has accumulated numerical drift); we raise
    RuntimeError in that case so callers notice.
    """
    if not matrices:
        raise ValueError("need at least one matrix")
    matrices = [np.asarray(g, dtype=complex) for g in matrices]
    d = matrices[0].shape[0]
    basis = su_basis(d)  # only computed once
    chi_sq_sum = 0.0
    for g in matrices:
        ad = adjoint_matrix(g, basis)       # (d²-1)×(d²-1) real; O(d⁴) memory
        chi = float(np.trace(ad))
        chi_sq_sum += chi * chi             # χ is real because Ad_g ∈ SO(d²-1)
    dim_float = chi_sq_sum / len(matrices)
    dim_int = int(round(dim_float))
    if abs(dim_int - dim_float) > integer_tol:
        raise RuntimeError(
            f"commutant dim {dim_float:.6f} is not an integer within tol "
            f"{integer_tol}. Input probably isn't a closed finite group — "
            "close it first with genGROUP.close_group or pass the full orbit."
        )
    return dim_int


def is_irreducible(matrices: Sequence[np.ndarray], *, integer_tol: float = 1e-6) -> bool:
    """True iff the adjoint representation of ⟨matrices⟩ on su(d) is irreducible.

    Equivalently: the common commutant is 1-dimensional (only scalar multiples
    of the identity commute with every Ad_g). Assumes the input is a closed
    finite group; see commutant_dimension docstring."""
    return commutant_dimension(matrices, integer_tol=integer_tol) == 1


def commutant_dimension_of_generators(
    matrices: Sequence[np.ndarray],
    *,
    rank_tol: float | None = None,
) -> int:
    """Dimension of the common commutant of {Ad_g : g ∈ matrices} on su(d),
    where `matrices` is a generating set --- not required to be closed.

    For a finite generating set S, Comm(Ad_⟨S⟩) = ∩_{g∈S} Comm(Ad_g)
    because M commuting with Ad_a and Ad_b implies M commutes with Ad_{ab}.
    So the commutant of the (possibly infinite) generated group equals
    the joint kernel over the finite generating set; this sidesteps the
    closed-group requirement of `commutant_dimension`.

    Implementation: for each g, build the (d²-1)²×(d²-1)² linear operator
    vec(M) ↦ vec(Ad_g · M - M · Ad_g), stack, and return (d²-1)² − rank.
    Memory is |S|·(d²-1)⁴ floats; fine through d=4.

    Used by `stages.check_extension` for the case where ⟨C, T⟩ is infinite
    (closure exceeded the classification cap) but we still want to know
    whether Ad_{⟨C,T⟩} is irreducible --- i.e., whether a reducible-Ad base
    group C has been lifted to universal by the extension T.
    """
    if not matrices:
        raise ValueError("need at least one matrix")
    mats = [np.asarray(g, dtype=complex) for g in matrices]
    d = mats[0].shape[0]
    basis = su_basis(d)
    D = d * d - 1
    I_D = np.eye(D)
    blocks = []
    for g in mats:
        ad = adjoint_matrix(g, basis)
        blocks.append(np.kron(ad, I_D) - np.kron(I_D, ad.T))
    A = np.vstack(blocks)
    if rank_tol is None:
        rank_tol = max(A.shape) * np.finfo(A.dtype).eps * np.linalg.norm(A, ord=2)
    rank = int(np.linalg.matrix_rank(A, tol=rank_tol))
    return D * D - rank


# ---------------------------------------------------------------------------
# Distance to centre Z(SU(d)) = {e^{2πik/d} I : 0 ≤ k < d}
# ---------------------------------------------------------------------------

def distance_to_center(g: np.ndarray) -> float:
    """Hilbert-Schmidt distance from g to the nearest centre element of SU(d).

    HS(A, B) := sqrt(tr((A-B)†(A-B))). For g ∈ SU(d) the centre is
    {e^{2πik/d} I : k = 0, ..., d-1}."""
    g = np.asarray(g, dtype=complex)
    d = g.shape[0]
    eye = np.eye(d, dtype=complex)
    def hs(diff: np.ndarray) -> float:
        return float(np.sqrt(np.real(np.trace(diff.conj().T @ diff))))
    return min(hs(g - np.exp(2j * np.pi * k / d) * eye) for k in range(d))


# ---------------------------------------------------------------------------
# Main entry point — structured result
# ---------------------------------------------------------------------------

Verdict = Literal["universal", "not_universal", "inconclusive"]


class SawickiResult(BaseModel):
    """Outcome of the Sawicki-Karnas universality test."""

    verdict: Verdict
    commutant_dim: int
    irreducible: bool
    min_distance_to_center: float
    has_near_center_element: bool        # some g with HS distance in (0, 1/√2]
    notes: str = ""


def check_universality(
    matrices: Sequence[np.ndarray],
    *,
    integer_tol: float = 1e-6,
    zero_tol: float = 1e-6,
) -> SawickiResult:
    """Run both halves of the Sawicki-Karnas criterion and classify.

    Verdicts:
        "universal"     — (A) irreducibility AND (B) some element has HS
                          distance to centre in (0, 1/√2].
        "not_universal" — (A) fails; closure of ⟨matrices⟩ is confined to a
                          proper subspace of SU(d).
        "inconclusive"  — (A) passes, (B) fails. The set may still be
                          universal (many real gate sets are); the caller
                          should fall back to a closure-termination test.
    """
    if not matrices:
        raise ValueError("need at least one matrix")
    cdim = commutant_dimension(matrices, integer_tol=integer_tol)
    irr = cdim == 1
    thresh = 1.0 / np.sqrt(2)
    dists = [distance_to_center(np.asarray(g)) for g in matrices]
    min_d = min(dists)
    near = any(zero_tol < d <= thresh for d in dists)

    if not irr:
        return SawickiResult(
            verdict="not_universal",
            commutant_dim=cdim,
            irreducible=False,
            min_distance_to_center=min_d,
            has_near_center_element=near,
            notes=f"Ad-commutant has dim {cdim} > 1 → reducible rep on su(d).",
        )
    if near:
        return SawickiResult(
            verdict="universal",
            commutant_dim=cdim,
            irreducible=True,
            min_distance_to_center=min_d,
            has_near_center_element=True,
            notes="Irreducibility + an element within HS distance 1/√2 of centre.",
        )
    return SawickiResult(
        verdict="inconclusive",
        commutant_dim=cdim,
        irreducible=True,
        min_distance_to_center=min_d,
        has_near_center_element=False,
        notes=(
            "Irreducibility holds but no element is within 1/√2 of the centre. "
            "The Sawicki-Karnas SUFFICIENT condition is strict; fall back to "
            "closure termination (close_group with generous max_size) to "
            "distinguish finite vs universal."
        ),
    )
