"""Curated catalog of magic-state distillation protocols.

Scope (per the agent design decision: known protocols only, research-needed
otherwise):
    * Bravyi-Kitaev |T⟩ via Reed-Muller [[15,1,3]] (the canonical protocol).
    * Bravyi-Haah triorthogonal qubit distillations including |CCZ⟩.
    * Qutrit [[20,7,2]]_3 triorthogonal distillation from Quantum 9, 1768 (2025).
    * Ternary Golay [[11,1,5]]_3 qutrit strange-state distillation.

We pair each protocol with a coarse `target_gate_family` tag. When matching
against an ExtensionSpec we use heuristics based on (spec.kind, spec.params,
dimension) — see `family_for_extension`.

For arbitrary extension gates not matching a tag, we return an empty list
and the supervisor logs a "no known protocol" outcome.
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
    # --- Qutrit ---
    DistillationRecord(
        protocol_name="Qutrit triorthogonal [[9m-k,k,2]]_3",
        target_gate_family="qutrit T",
        qudit_dim=3,
        code_name="Qutrit triorthogonal [[20,7,2]]_3",
        yield_parameter=1.51,
        reference="Low-overhead qutrit MSD, Quantum 9, 1768 (2025)",
        notes=(
            "Family [[9m-k,k,2]]_3 triorthogonal qutrit codes. "
            "The [[20,7,2]]_3 member has γ=1.51, beating all qubit triorthogonal "
            "codes with n < ~300 physical qudits. Applicable to the Howard-Vala T family."
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

def family_for_extension(ext_kind: str, ext_params: dict, qudit_dim: int) -> str | None:
    """Map an ExtensionSpec to a coarse distillation-catalog family tag.

    Heuristic — does not claim equivalence; just says "this spec *might* be
    distilled by protocols tagged with this family". Returns None when no
    known family applies.
    """
    if ext_kind == "howard_vala":
        return "qubit T" if qudit_dim == 2 else f"qutrit T" if qudit_dim == 3 else None
    if ext_kind == "angle":
        # angle ≈ π/4 on qubit = canonical T.
        theta = float(ext_params.get("theta", 0.0))
        if qudit_dim == 2 and abs(abs(theta) - (3.141592653589793 / 4)) < 1e-6:
            return "qubit T"
        return None
    # 'mat', 'angles', 'rnd' are too generic to map cleanly without more info.
    return None


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
    """Main Stage 5 entry point. Returns (protocols, notes)."""
    family = family_for_extension(ext_kind, ext_params, qudit_dim)
    if family is None:
        return [], (
            f"No catalog-mapped distillation family for "
            f"(kind={ext_kind!r}, d={qudit_dim}). "
            "Research may be needed — consult Error Correction Zoo magic-state list."
        )
    hits = protocols_for_family(family, qudit_dim)
    if not hits:
        return [], (
            f"Mapped to family {family!r} but no protocols registered for "
            f"d={qudit_dim}. Research may be needed."
        )
    note = (
        f"{len(hits)} catalog protocol(s) for family {family!r} at d={qudit_dim}."
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
