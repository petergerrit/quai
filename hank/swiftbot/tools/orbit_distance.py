"""Clifford-orbit distance: how close is a candidate non-Clifford unitary U
to the double-coset C · D · C of the base group C and a rational-angle
diagonal D?

The pipeline uses this to annotate rnd winners with a fault-tolerance hint:
if U is near C · D_rational · C for some D with algebraic-angle phases, its
magic state can in principle be prepared via Clifford-conjugated distillation
of D. Large residual → no known direct distillation route; approximate
synthesis (Ross-Selinger / Solovay-Kitaev) is the only path.

Concretely, for U ∈ SU(d) and finite subgroup C ⊂ SU(d), we compute
    residual(U, C) = min_{g, h ∈ C}  || U − g · diag(g† U h†)_rounded · h ||_F
where the inner diagonal is the projection of g† U h† onto its diagonal
part, with entries optionally snapped to the nearest rational angle over a
set of candidate denominators.

Runtime is O(|C|² · d²); tractable through |C| ~ 1000 for d ≤ 4.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


@dataclass
class OrbitDistanceResult:
    """One invocation of clifford_orbit_distance()."""
    residual: float                       # Frobenius distance ||U - g D h||
    g_index: int                          # index into C
    h_index: int                          # index into C
    D_phases: tuple[float, ...]           # diagonal entries' phases (radians)
    D_phases_over_pi: tuple[float, ...]   # phases normalised by π
    best_rational_denoms: tuple[int, ...] # nearest rational denominator per phase
    best_rational_residues: tuple[float, ...]  # how far each phase is off rational
    approximately_distillable: bool       # residual below threshold AND all phases rational

    def to_dict(self) -> dict:
        return asdict(self)


def _nearest_rational(
    phi_over_pi: float,
    denoms: Sequence[int],
    tol: float = 5e-3,
) -> tuple[int, float]:
    """Find the denominator in `denoms` with smallest integer-distance for
    phi / π. Returns (denom, residue) where residue = φ·denom − round(φ·denom).
    When no denominator reaches |residue| < tol, returns (0, min |residue|)."""
    best_denom, best_residue = 0, float('inf')
    for k in denoms:
        r = phi_over_pi * k - round(phi_over_pi * k)
        if abs(r) < abs(best_residue):
            best_denom, best_residue = k, r
    return best_denom, best_residue


def clifford_orbit_distance(
    U: np.ndarray,
    C: Sequence[np.ndarray],
    *,
    rational_denoms: Sequence[int] = (3, 4, 6, 8, 9, 12, 16, 18, 24, 36),
    rational_tol: float = 5e-3,
    residual_threshold: float = 0.05,
) -> OrbitDistanceResult:
    """Compute the Frobenius distance from U to the double-coset C·diag·C of
    the finite group C, with the diagonal entries optionally snapped to the
    nearest rational-angle class.

    Args:
        U: d×d unitary, assumed in SU(d) (det 1). Not required to be in any
            subgroup of SU(d).
        C: finite group elements, list/array of d×d unitaries.
        rational_denoms: candidate denominators k for testing phase/π ≈ n/k.
        rational_tol: residue threshold below which a phase is declared
            rational on that denominator.
        residual_threshold: Frobenius-distance ceiling below which U is
            flagged as approximately-distillable.

    Returns:
        OrbitDistanceResult. For a generic Haar-sampled U this typically has
        residual ~0.5 × √d on SU(d), well above the threshold; for U lying
        in the double-coset of a rational-angle diagonal (e.g. an element
        obtained as g · P(2π/9) · h for g, h ∈ C), the residual collapses
        to numerical noise.
    """
    U = np.asarray(U, dtype=complex)
    d = U.shape[0]
    if U.shape != (d, d):
        raise ValueError(f"U must be square; got {U.shape}")
    C = [np.asarray(c, dtype=complex) for c in C]
    nC = len(C)

    # Precompute g† U for each g
    gt_U = np.stack([g.conj().T @ U for g in C], axis=0)  # (nC, d, d)
    # Precompute h† for each h
    h_T = np.stack([h.conj().T for h in C], axis=0)       # (nC, d, d)

    best_off = np.inf
    best = (0, 0, None)
    # Vectorise inner h loop: for each g, compute Y[h] = gt_U[g] @ h_T[h] and its
    # off-diagonal Frobenius norm across h.
    for gi in range(nC):
        Y = gt_U[gi] @ h_T                     # (nC, d, d)
        diag_Y = np.diagonal(Y, axis1=1, axis2=2)  # (nC, d)
        # Off-diag Frobenius: ||Y||² - |diag|² sum
        total_sq = np.einsum('nij,nij->n', Y.conj(), Y).real
        diag_sq = np.einsum('nj,nj->n', diag_Y.conj(), diag_Y).real
        off_sq = total_sq - diag_sq
        hj = int(np.argmin(off_sq))
        off = float(np.sqrt(off_sq[hj]))
        if off < best_off:
            best_off = off
            best = (gi, hj, diag_Y[hj])

    gi, hj, D_diag = best
    D_diag = np.asarray(D_diag, dtype=complex)
    # Normalise: the optimal D for minimum ||U − g D h|| (with fixed g, h) is
    # the diagonal of g† U h†, regardless of D's element magnitudes. Above we
    # projected by copying the diagonal straight out.
    phases = np.angle(D_diag)
    phases_over_pi = phases / np.pi

    # Rational-angle fit per phase
    best_denoms: list[int] = []
    best_residues: list[float] = []
    all_rational = True
    for phi_pi in phases_over_pi:
        denom, residue = _nearest_rational(float(phi_pi), rational_denoms, rational_tol)
        best_denoms.append(denom)
        best_residues.append(residue)
        if denom == 0 or abs(residue) > rational_tol:
            all_rational = False

    approximately_distillable = (best_off <= residual_threshold) and all_rational

    return OrbitDistanceResult(
        residual=best_off,
        g_index=gi,
        h_index=hj,
        D_phases=tuple(phases.tolist()),
        D_phases_over_pi=tuple(phases_over_pi.tolist()),
        best_rational_denoms=tuple(best_denoms),
        best_rational_residues=tuple(best_residues),
        approximately_distillable=approximately_distillable,
    )
