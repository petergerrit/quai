"""Curated catalog of quantum error-correcting codes + transversal-gate groups.

Scope (per the agent design decision: tractable codes only):
    * Qubit stabilizer codes with known transversal-gate classification
      (Anderson-Jochym-O'Connor, arXiv:1409.8320; Grier-Schaeffer).
    * Kubischta's exotic non-stabilizer codes with transversal 2I / 2O.
    * Qutrit triorthogonal codes from the 2025 Quantum paper for qutrit MSD.
    * One or two ternary Golay entries.

For base groups not covered here we return an empty list with a pointer to
errorcorrectionzoo.org — the caller (stage 4 in the supervisor) annotates
those with `research_needed=True`.

Cross-validation: the `transversal_groups` strings MUST be a subset of
swiftbot.tools.groups.REGISTRY keys; tested in tests/test_codes.py.
"""
from __future__ import annotations

from typing import Iterable

from swiftbot.state import CodeRecord

# ---------------------------------------------------------------------------
# Curated catalog
# ---------------------------------------------------------------------------

# Each entry cites a concrete paper. We stay conservative: only include
# transversal-group assignments that the literature states explicitly.
_CODES: list[CodeRecord] = [
    # --- Qubit stabilizer codes ---
    CodeRecord(
        name="Steane [[7,1,3]]",
        n=7, k=1, distance=3, qudit_dim=2,
        transversal_groups=["clifford"],
        stabilizer=True,
        reference="Steane, quant-ph/9601029",
        notes="All Cliffords transversal; first CSS triple-error-detecting code.",
    ),
    CodeRecord(
        name="5-qubit perfect [[5,1,3]]",
        n=5, k=1, distance=3, qudit_dim=2,
        transversal_groups=[],   # single-qubit Clifford transversal group on [[5,1,3]] is cyclic, not all of Clifford
        stabilizer=True,
        reference="Laflamme-Miquel-Paz-Zurek, PRL 77, 198 (1996)",
        notes=(
            "Single-qubit transversal group is cyclic (Gottesman); not registered "
            "in SWIFTbot. Included for reference — not a match for 'clifford'."
        ),
    ),
    CodeRecord(
        name="Reed-Muller [[15,1,3]]",
        n=15, k=1, distance=3, qudit_dim=2,
        transversal_groups=["clifford"],
        stabilizer=True,
        reference="Knill-Laflamme-Zurek, Science 279 (1998); Bravyi-Kitaev 2005",
        notes=(
            "Transversal T-gate; union with the transversal-Clifford subgroup of "
            "other RM codes gives Clifford+T. Hosts Bravyi-Kitaev 15-to-1 distillation."
        ),
    ),
    CodeRecord(
        name="Bravyi-Haah [[49,1,5]] triorthogonal",
        n=49, k=1, distance=5, qudit_dim=2,
        transversal_groups=["clifford"],
        stabilizer=True,
        reference="Bravyi-Haah, PRA 86, 052329 (2012)",
        notes="Triorthogonal; transversal T and CCZ via Reed-Muller punctures.",
    ),
    # --- Exotic / non-stabilizer qubit codes ---
    CodeRecord(
        name="Kubischta 2I-icosahedral family ((7,2,3))",
        n=7, k=2, distance=3, qudit_dim=2,
        transversal_groups=["BI"],
        stabilizer=False,
        reference="Kubischta & Teixeira, PRL 131, 240601 (2023) [arXiv:2305.07023]",
        notes="Nonadditive quantum code with transversal binary icosahedral group 2I = 120.",
    ),
    CodeRecord(
        name="Kubischta 2O family",
        n=None, k=None, distance=None, qudit_dim=2,
        transversal_groups=["BO"],
        stabilizer=False,
        reference="Kubischta & Teixeira, arXiv:2305.07023 (and related work)",
        notes=(
            "Family admitting transversal binary octahedral 2O = 48 action. "
            "Parameters (n,k,d) family-dependent; see paper for specific instances."
        ),
        research_needed=True,
    ),
    # --- Qutrit stabilizer / triorthogonal ---
    CodeRecord(
        name="Qutrit triorthogonal [[20,7,2]]_3",
        n=20, k=7, distance=2, qudit_dim=3,
        transversal_groups=[],
        stabilizer=True,
        reference="Low-overhead qutrit MSD, Quantum 9, 1768 (2025)",
        notes=(
            "Family [[9m-k,k,2]]_3; this member has yield γ ≈ 1.51. "
            "Hosts qutrit T (Howard-Vala / Campbell) magic-state distillation. "
            "No full transversal-group classification for qutrit stabilizer codes yet."
        ),
    ),
    CodeRecord(
        name="Ternary Golay [[11,1,5]]_3",
        n=11, k=1, distance=5, qudit_dim=3,
        transversal_groups=[],
        stabilizer=True,
        reference="Magic state distillation with the ternary Golay code, RSPA (2020)",
        notes="Distills the 'strange' qutrit magic state; no fully classified transversal group.",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_all_codes() -> list[CodeRecord]:
    """Return a shallow copy of the full curated catalog."""
    return list(_CODES)


def codes_with_transversal_group(group_name: str) -> list[CodeRecord]:
    """All curated codes that admit `group_name` transversally."""
    return [c for c in _CODES if group_name in c.transversal_groups]


def codes_for_dim(qudit_dim: int) -> list[CodeRecord]:
    """All curated codes for a given qudit dimension."""
    return [c for c in _CODES if c.qudit_dim == qudit_dim]


def codes_for_group(group_name: str) -> tuple[list[CodeRecord], str]:
    """Look up codes compatible with `group_name` (Stage 4 entry point).

    Returns a (codes, notes) pair. When nothing matches, `codes` is empty
    and `notes` contains a pointer to errorcorrectionzoo.org.
    """
    hits = codes_with_transversal_group(group_name)
    if hits:
        note = f"{len(hits)} curated code(s) match base group {group_name!r}."
        return hits, note
    note = (
        f"No curated code with transversal group {group_name!r}. "
        "Research may be needed — start at https://errorcorrectionzoo.org/c/qudit_stabilizer "
        "and https://errorcorrectionzoo.org/c/qubit_stabilizer."
    )
    return [], note


def validate_registry_names(registry: Iterable[str]) -> list[str]:
    """Return any `transversal_groups` strings that are NOT in the given
    registry — used as a schema-integrity test."""
    registered = set(registry)
    bad: list[str] = []
    for c in _CODES:
        for g in c.transversal_groups:
            if g not in registered:
                bad.append(f"{c.name} → {g!r} (not in registry)")
    return bad
