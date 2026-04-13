"""Lamm-series lattice QFT target families.

These are derived directly from the primitive-gate papers (BT/BO/Σ(36×3)/
Σ(72×3)). For each group:

    * Discrete part: R_Z phases arising in the qubit-encoded U_F (non-Clifford
      cyclotomic roots of unity).
    * Parametric part: R_Z(k · θ) for k ∈ {1, 3, 5, 9, 15} with θ ∝ β·Δt
      (continuous in the lattice coupling and Trotter step).

References:
    * Gustafson et al., "Primitive Quantum Gates for Σ(36×3)", arXiv:2405.05973.
    * Osorio Perez et al., "Primitive Quantum Gates for Σ(72×3)", arXiv:2511.17437.
    * Murairi et al., "Highly-efficient QFT for nonabelian groups",
      arXiv:2408.00075 (Sec IV.3, Eq 28 / Eq 61).
"""
from __future__ import annotations

import numpy as np

from swiftbot.targets import TargetFamily, register


_CONTINUOUS_K = (1, 3, 5, 9, 15)


def _rz(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2), 0.0],
         [0.0, np.exp(1j * phi / 2)]],
        dtype=complex,
    )


def _discrete_9th_roots() -> list[tuple[str, np.ndarray]]:
    """8 discrete R_Z phases from Σ(36×3) U_F: 2π·k/9 for k = 1..8."""
    return [
        (f"R_Z(2π·{k}/9)", _rz(2 * np.pi * k / 9))
        for k in range(1, 9)
    ]


def _discrete_18th_roots_new() -> list[tuple[str, np.ndarray]]:
    """18th-root phases unique to Σ(72×3) (i.e. not already 9th roots).

    Σ(72×3) irreps inherit 9th-root structure from Σ(36×3) and add 6th-root
    structure from the V₂ extension. Lcm(9, 6) = 18 → phases in ℤ[ω_18].
    We list only the genuinely new ones: 2π·k/18 for k with 18/gcd(18,k) != 9.
    """
    out: list[tuple[str, np.ndarray]] = []
    for k in range(1, 18):
        # Skip any k that reduces to a 9th-root (i.e. 2k/18 = j/9 for some j)
        # ⇔ k is even AND k/2 ∈ {1,...,8}. Exclude k ∈ {2, 4, 6, 8, 10, 12, 14, 16}.
        if k in {2, 4, 6, 8, 10, 12, 14, 16}:
            continue
        out.append((f"R_Z(2π·{k}/18)", _rz(2 * np.pi * k / 18)))
    return out


def _parametric_continuous(theta: float) -> list[tuple[str, np.ndarray]]:
    """5 R_Z(k·θ) rotations from U_Tr, per Eq 28 of 2405.05973."""
    return [
        (f"R_Z({k}·θ)", _rz(k * theta))
        for k in _CONTINUOUS_K
    ]


register(TargetFamily(
    name="lamm_sigma36",
    description=(
        "Σ(36×3) lattice QFT target: 8 discrete 9th-root phases from U_F "
        "(Eq 28 of arXiv:2405.05973) + 5 continuous R_Z(k·θ) rotations from "
        "U_Tr (k ∈ {1,3,5,9,15}, θ = c·β·Δt)."
    ),
    qudit_dim=2,
    discrete=_discrete_9th_roots(),
    parametric=_parametric_continuous,
    reference="arXiv:2405.05973, arXiv:2408.00075",
))


register(TargetFamily(
    name="lamm_sigma72",
    description=(
        "Σ(72×3) lattice QFT target: Σ(36×3) phases PLUS the 18th-root "
        "phases that Σ(72×3)'s Δ(54) extension introduces (Sec V of "
        "arXiv:2408.00075). U_Tr is identical to Σ(36×3)."
    ),
    qudit_dim=2,
    discrete=_discrete_9th_roots() + _discrete_18th_roots_new(),
    parametric=_parametric_continuous,
    reference="arXiv:2511.17437, arXiv:2408.00075",
))
