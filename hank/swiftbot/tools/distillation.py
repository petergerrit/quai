"""Curated catalog of magic-state distillation protocols.

Scope (per the agent design decision: known protocols only, research-needed
otherwise):
    * Bravyi-Kitaev |T⟩ via Reed-Muller [[15,1,3]] (the canonical protocol).
    * Bravyi-Haah triorthogonal qubit distillations including |CCZ⟩.
    * Programmable qubit MSD for |ψ(θ)⟩ at arbitrary θ (Duclos-Cianci + Poulin
      2015; Campbell + O'Gorman 2016). Built on a Clifford+T substrate — these
      do not give a new transversal primitive, but they DO give a fault-tolerant
      route to any Z-rotation angle, including those outside the Clifford
      hierarchy where Anderson-Jochym-O'Connor forbids a direct stabilizer-code
      transversal implementation.
    * Qutrit [[20,7,2]]_3 triorthogonal distillation from Quantum 9, 1768 (2025).
    * Ternary Golay [[11,1,5]]_3 qutrit strange-state distillation.

We pair each protocol with a coarse `target_gate_family` tag. When matching
against an ExtensionSpec we use heuristics based on (spec.kind, spec.params,
dimension) — see `family_for_extension`.

For arbitrary extension gates not matching a tag, we return an empty list
and the supervisor logs a "no known protocol" outcome. For gates that are
provably outside any qubit stabilizer-code transversal set (Anderson-Jochym-
O'Connor 2014), `protocols_for_extension` annotates the note accordingly.
"""
from __future__ import annotations

from typing import Iterable

from swiftbot.state import DistillationRecord

# ---------------------------------------------------------------------------
# Curated catalog
# ---------------------------------------------------------------------------

_PROTOCOLS: list[DistillationRecord] = [
    # --- Qubit ---
    DistillationRecord(
        protocol_name="Bravyi-Kitaev 15-to-1",
        target_gate_family="qubit T",
        qudit_dim=2,
        code_name="Reed-Muller [[15,1,3]]",
        yield_parameter=None,
        reference="Bravyi-Kitaev, PRA 71, 022316 (2005)",
        notes="Canonical |T⟩ distillation; produces one high-fidelity T-state from 15 noisy ones.",
    ),
    DistillationRecord(
        protocol_name="Bravyi-Haah triorthogonal",
        target_gate_family="qubit T",
        qudit_dim=2,
        code_name="Bravyi-Haah [[49,1,5]] triorthogonal",
        yield_parameter=None,
        reference="Bravyi-Haah, PRA 86, 052329 (2012)",
        notes="Higher-distance triorthogonal distillation; multiple [[n,1,d]] constructions.",
    ),
    DistillationRecord(
        protocol_name="Bravyi-Haah CCZ",
        target_gate_family="qubit CCZ",
        qudit_dim=2,
        code_name="Bravyi-Haah [[49,1,5]] triorthogonal",
        yield_parameter=None,
        reference="Bravyi-Haah, PRA 86, 052329 (2012)",
        notes="Distills |CCZ⟩ magic state; Clifford+CCZ universal for qubits.",
    ),
    # --- Programmable qubit MSD: arbitrary-θ Z-rotation on a Clifford+T substrate ---
    DistillationRecord(
        protocol_name="Duclos-Cianci + Poulin complex-gate distillation",
        target_gate_family="qubit Z-rotation (programmable)",
        qudit_dim=2,
        code_name="Clifford+T substrate",
        yield_parameter=None,
        reference="Duclos-Cianci & Poulin, PRA 91, 042315 (2015) [arXiv:1403.5280]",
        notes=(
            "Programmable distillation of |ψ(θ)⟩ for operator-specified θ via "
            "complex-gate teleportation. Does NOT give a new transversal primitive "
            "(Anderson-Jochym-O'Connor rules that out for any θ outside the "
            "Clifford hierarchy); runs on a Clifford+T fault-tolerant substrate. "
            "Overhead scales with the Ross-Selinger T-count of P(θ) and is "
            "comparable to direct Clifford+T synthesis for θ ~ π/4."
        ),
    ),
    DistillationRecord(
        protocol_name="Campbell + O'Gorman small-angle rotations",
        target_gate_family="qubit Z-rotation (programmable)",
        qudit_dim=2,
        code_name="Clifford+T substrate",
        yield_parameter=None,
        reference="Campbell & O'Gorman, Quantum Sci. Technol. 1, 015007 (2016) [arXiv:1603.04230]",
        notes=(
            "Refinement of Duclos-Cianci/Poulin: tighter analysis and a dilution "
            "protocol that becomes significantly better than Ross-Selinger for "
            "θ << π/4. At θ ≈ π/4 (e.g. 2π/9 ≈ 40°) the overhead is comparable "
            "to Clifford+T synthesis. Same Clifford+T substrate."
        ),
    ),
    DistillationRecord(
        protocol_name="Campbell-Howard parity-checker (pre-distilled)",
        target_gate_family="qubit Z-rotation (programmable)",
        qudit_dim=2,
        code_name="Clifford+T substrate",
        yield_parameter=None,
        reference="Campbell & Howard, arXiv:1709.02214 (2017)",
        notes=(
            "Two-step protocol: pre-distillation of multi-qubit resource states, "
            "followed by non-Pauli parity checking. Enables direct injection of "
            "single-qubit axial rotations without subsequent gate-synthesis, with "
            "quadratic error reduction. Clifford+T substrate, so AJOC exclusion of "
            "a direct stabilizer-code transversal implementation still applies — "
            "this is a programmable-MSD route, not an AJOC loophole."
        ),
    ),
    # --- Non-stabilizer (PI code) code-switching: a second AJOC loophole ---
    DistillationRecord(
        protocol_name="Ouyang-Jing-Brennen PI code-switching",
        target_gate_family="qubit rational-angle (non-stabilizer code-switching)",
        qudit_dim=2,
        code_name="Kubischta-Teixeira (2b+3)-qubit PI family",
        yield_parameter=None,
        reference="Ouyang, Jing & Brennen, arXiv:2411.13142 (2024)",
        notes=(
            "Not MSD: measurement-free code-switching between a stabilizer code "
            "(transversal Cliffords) and a permutation-invariant (PI) code with "
            "transversal Z(πg/b) for rational g/b, glued by CNOTs through a "
            "catalytic bosonic mode. Sidesteps the Anderson-Jochym-O'Connor no-go "
            "entirely — AJOC forbids transversal P(θ) on stabilizer codes, but "
            "PI codes are non-stabilizer. Provides a second fault-tolerant route "
            "to arbitrary rational-angle qubit rotations alongside programmable "
            "MSD (Duclos-Cianci-Poulin, Campbell-O'Gorman, Campbell-Howard). "
            "Code-switching is the paper's consumption pattern; state injection "
            "on the same PI-encoded magic state |M(θ)⟩ = Z(θ)|+⟩_L is a viable "
            "alternative (but requires measurement-based inter-code conversion)."
        ),
    ),
    # --- Qutrit ---
    DistillationRecord(
        protocol_name="Qutrit triorthogonal [[9m-k,k,2]]_3",
        target_gate_family="qutrit T",
        qudit_dim=3,
        code_name="Qutrit triorthogonal [[20,7,2]]_3",
        yield_parameter=1.51,
        reference="Prakash & Saha, Quantum 9, 1768 (2025) [arXiv:2403.06228]",
        notes=(
            "Family [[9m-k,k,2]]_3 triorthogonal qutrit codes. "
            "The [[20,7,2]]_3 member has γ=1.51, beating all qubit triorthogonal "
            "codes with n < ~300 physical qudits. Applicable to the Howard-Vala T family."
        ),
    ),
    # --- Higher-dimensional qudit constant-overhead CCZ distillation ---
    DistillationRecord(
        protocol_name="Cervia-Lamm-Liu-Murairi-Zhu AG-qudit CCZ",
        target_gate_family="qudit CCZ (q=64)",
        qudit_dim=64,
        code_name="Asymptotically-good triorthogonal codes on F_{2^{2m}}",
        yield_parameter=0.0,  # constant-space overhead (γ → 0 asymptotically)
        reference="Cervia, Lamm, Liu, Murairi & Zhu, arXiv:2512.21874 (2025)",
        notes=(
            "Constant-space-overhead magic-state distillation on qudits of "
            "dimension q = 2^(2m), m ≥ 3 (i.e. q ≥ 64). First triorthogonal "
            "code family on F_{2^{2m}} that beats the Tsfasman-Vlăduţ-Zink "
            "bound. Distills |CCZ⟩ at constant space overhead, far better "
            "than the γ > 0 scaling of qubit triorthogonal codes. A "
            "constant-depth Clifford circuit converts the qudit |CCZ⟩ down "
            "to qubit |CCZ⟩ via ≤9 measurements, so the construction is "
            "usable as a qubit-MSD drop-in as well. Builds on and "
            "generalises Golowich & Guruswami (STOC 2024, arXiv:2408.09254) "
            "and Nguyen (STOC 2025) for the binary case, and Krishna & "
            "Tillich (PRL 123, 070507) for the low-overhead qudit-MSD "
            "programme more broadly."
        ),
    ),
    DistillationRecord(
        protocol_name="Ternary Golay strange-state",
        target_gate_family="qutrit strange",
        qudit_dim=3,
        code_name="Ternary Golay [[11,1,5]]_3",
        yield_parameter=None,
        reference="Magic state distillation with the ternary Golay code, RSPA (2020)",
        notes="Distills the ‘strange’ qutrit magic state; high threshold but low yield.",
    ),
]


# ---------------------------------------------------------------------------
# Extension-spec → gate family heuristic
# ---------------------------------------------------------------------------

import math


def family_for_extension(ext_kind: str, ext_params: dict, qudit_dim: int) -> str | None:
    """Map an ExtensionSpec to a coarse distillation-catalog family tag.

    Heuristic — does not claim equivalence; just says "this spec *might* be
    distilled by protocols tagged with this family". Returns None when no
    known family applies.

    At d=2, any `angle` extension not at θ = π/4 routes to the
    "qubit Z-rotation (programmable)" family: Anderson-Jochym-O'Connor
    forbids a direct stabilizer-code transversal implementation, so the
    fault-tolerant route is programmable MSD (Duclos-Cianci-Poulin,
    Campbell-O'Gorman) on a Clifford+T substrate.
    """
    if ext_kind == "howard_vala":
        return "qubit T" if qudit_dim == 2 else ("qutrit T" if qudit_dim == 3 else None)
    if ext_kind == "angle":
        theta = float(ext_params.get("theta", 0.0))
        if qudit_dim == 2:
            if abs(abs(theta) - math.pi / 4) < 1e-6:
                return "qubit T"  # canonical T — direct distillation.
            return "qubit Z-rotation (programmable)"  # programmable MSD on Clifford+T.
        return None
    if ext_kind == "angles" and qudit_dim == 2:
        # Any qubit angles spec reduces to a product of single-qubit Z rotations.
        return "qubit Z-rotation (programmable)"
    # 'mat', 'rnd' are too generic: a generic Haar unitary is not exactly
    # synthesisable and therefore not distillable in the standard MSD sense.
    return None


_NON_STABILIZER_BYPASS = (
    " Note: this is a STABILIZER-CODE no-go only. Non-stabilizer qubit codes "
    "(Pollatsek-Ruskai and Kubischta-Teixeira permutation-invariant codes, "
    "Denys-Leverrier covariant-encoding 2T/2O codes, Kubischta 2I) can host "
    "transversal non-Clifford gates directly. See protocol "
    "'Ouyang-Jing-Brennen PI code-switching' for a fault-tolerant route to "
    "arbitrary rational-angle Z-rotations that sidesteps AJOC."
)


def ajoc_excluded(ext_kind: str, ext_params: dict, qudit_dim: int) -> tuple[bool, str]:
    """Is this extension provably outside every qubit STABILIZER-code's
    transversal set (Anderson-Jochym-O'Connor 2014 [arXiv:1409.8320])?

    Returns (excluded, reason_string). At d=2, any non-Clifford-hierarchy gate
    is excluded; that includes every diagonal phase at a rational multiple of
    π with an odd denominator factor (e.g. 2π/9, 2π/5, π/9, π/18) and every
    Haar-random generator. The canonical Clifford+T at π/4 is at level 3 of
    the hierarchy and is NOT excluded.

    IMPORTANT: the AJOC theorem applies to STABILIZER codes only. Non-stabilizer
    qubit codes (e.g. Pollatsek-Ruskai and Kubischta-Teixeira PI codes,
    Denys-Leverrier covariant codes) can host transversal non-Clifford gates
    directly; the `reason_string` notes this loophole so callers don't
    over-claim impossibility.

    At d>2 we have no analogous classification theorem (see Campbell et al.);
    this function returns (False, "no known qudit classification") for d>=3.
    """
    if qudit_dim >= 3:
        return False, "no classification analogue of Anderson-Jochym-O'Connor for d≥3"
    if qudit_dim == 2:
        if ext_kind == "angle":
            theta = float(ext_params.get("theta", 0.0))
            if abs(abs(theta) - math.pi / 4) < 1e-6:
                return False, "θ = π/4 is in the Clifford hierarchy (canonical T)"
            return True, (
                "non-Clifford-hierarchy Z-rotation; Anderson-Jochym-O'Connor "
                "forbids any qubit STABILIZER code with transversal P(θ) at this angle."
                + _NON_STABILIZER_BYPASS
            )
        if ext_kind == "rnd":
            return True, (
                "generic Haar extension; not exactly synthesisable and not "
                "a Clifford-hierarchy element." + _NON_STABILIZER_BYPASS
            )
        if ext_kind == "mat":
            return True, (
                "explicit SU(2) matrix; unless it coincides with a Clifford-hierarchy "
                "element, Anderson-Jochym-O'Connor excludes a direct transversal "
                "STABILIZER-code distillation." + _NON_STABILIZER_BYPASS
            )
    return False, "no exclusion determined"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_all_protocols() -> list[DistillationRecord]:
    return list(_PROTOCOLS)


def protocols_for_family(family: str, qudit_dim: int) -> list[DistillationRecord]:
    return [
        p for p in _PROTOCOLS
        if p.target_gate_family == family and p.qudit_dim == qudit_dim
    ]


def protocols_for_extension(
    ext_kind: str, ext_params: dict, qudit_dim: int,
) -> tuple[list[DistillationRecord], str]:
    """Main Stage 5 entry point. Returns (protocols, notes).

    When `ajoc_excluded` reports direct stabilizer-code distillation is ruled
    out (e.g., P(2π/9) at d=2), the note flags that explicitly so downstream
    supervisors and paper authors don't over-claim a direct distillation route.
    """
    family = family_for_extension(ext_kind, ext_params, qudit_dim)
    excluded, reason = ajoc_excluded(ext_kind, ext_params, qudit_dim)
    ajoc_prefix = (
        f"Anderson-Jochym-O'Connor exclusion applies: {reason}. "
        if excluded else ""
    )

    if family is None:
        return [], (
            f"{ajoc_prefix}No catalog-mapped distillation family for "
            f"(kind={ext_kind!r}, d={qudit_dim}). "
            "Research may be needed — consult Error Correction Zoo magic-state list."
        )
    hits = protocols_for_family(family, qudit_dim)
    if not hits:
        return [], (
            f"{ajoc_prefix}Mapped to family {family!r} but no protocols registered for "
            f"d={qudit_dim}. Research may be needed."
        )
    note = (
        f"{ajoc_prefix}{len(hits)} catalog protocol(s) for family {family!r} at d={qudit_dim}."
    )
    return hits, note


def validate_catalog() -> list[str]:
    """Sanity-check the catalog for obvious bugs. Returns problem strings."""
    bad: list[str] = []
    for p in _PROTOCOLS:
        if p.qudit_dim < 2:
            bad.append(f"{p.protocol_name}: qudit_dim={p.qudit_dim}")
        if not p.code_name:
            bad.append(f"{p.protocol_name}: empty code_name")
        if not p.reference:
            bad.append(f"{p.protocol_name}: missing reference")
    return bad
