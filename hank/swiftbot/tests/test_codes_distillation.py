"""Tests for the Stage-4 (codes) and Stage-5 (distillation) catalogs.

Focus: schema integrity (transversal-group references are valid registry
names), lookup-by-group semantics, and the extension-family heuristic.
"""
from __future__ import annotations

import math

from swiftbot.tools import codes, distillation
from swiftbot.tools import groups as gmod


# ---------------------------------------------------------------------------
# codes.py
# ---------------------------------------------------------------------------

def test_catalog_transversal_groups_all_registered() -> None:
    """Every transversal_groups entry in the code catalog must refer to a
    SWIFTbot registry name — catches typos and stale references."""
    bad = codes.validate_registry_names(gmod.REGISTRY)
    assert not bad, "unregistered transversal groups:\n" + "\n".join(bad)


def test_codes_for_clifford_includes_steane_and_rm() -> None:
    hits, note = codes.codes_for_group("clifford")
    names = {c.name for c in hits}
    assert "Steane [[7,1,3]]" in names
    assert "Reed-Muller [[15,1,3]]" in names
    assert "curated" in note.lower()


def test_codes_for_BI_includes_kubischta() -> None:
    hits, _ = codes.codes_for_group("BI")
    assert any("Kubischta" in c.name for c in hits)


def test_codes_for_unknown_group_returns_research_needed_note() -> None:
    hits, note = codes.codes_for_group("s720x4")
    assert hits == []
    assert "Research may be needed" in note
    assert "errorcorrectionzoo" in note


def test_codes_for_dim_filters() -> None:
    qubit_codes = codes.codes_for_dim(2)
    qutrit_codes = codes.codes_for_dim(3)
    assert all(c.qudit_dim == 2 for c in qubit_codes)
    assert all(c.qudit_dim == 3 for c in qutrit_codes)
    assert any("Steane" in c.name for c in qubit_codes)
    assert any("Golay" in c.name for c in qutrit_codes)


# ---------------------------------------------------------------------------
# distillation.py
# ---------------------------------------------------------------------------

def test_catalog_basic_sanity() -> None:
    bad = distillation.validate_catalog()
    assert not bad, "catalog errors:\n" + "\n".join(bad)


def test_family_for_extension_howard_vala() -> None:
    assert distillation.family_for_extension("howard_vala", {}, 2) == "qubit T"
    assert distillation.family_for_extension("howard_vala", {}, 3) == "qutrit T"
    assert distillation.family_for_extension("howard_vala", {}, 5) is None


def test_family_for_extension_angle_matches_pi_4() -> None:
    assert distillation.family_for_extension("angle", {"theta": math.pi / 4}, 2) == "qubit T"
    assert distillation.family_for_extension("angle", {"theta": math.pi / 4}, 3) is None


def test_family_for_extension_non_pi4_angle_routes_to_programmable() -> None:
    """Any θ != π/4 at d=2 should map to the programmable-MSD family."""
    assert distillation.family_for_extension(
        "angle", {"theta": 2 * math.pi / 9}, 2
    ) == "qubit Z-rotation (programmable)"
    assert distillation.family_for_extension(
        "angle", {"theta": math.pi / 8}, 2
    ) == "qubit Z-rotation (programmable)"
    assert distillation.family_for_extension(
        "angles", {"phases": [0.1, -0.1]}, 2
    ) == "qubit Z-rotation (programmable)"


def test_protocols_for_qubit_T_include_bravyi_kitaev() -> None:
    hits, note = distillation.protocols_for_extension("angle", {"theta": math.pi / 4}, 2)
    assert any(p.protocol_name.startswith("Bravyi-Kitaev") for p in hits)
    assert any("Reed-Muller" in p.code_name for p in hits)
    assert "family" in note


def test_protocols_for_qutrit_T_include_triorthogonal() -> None:
    hits, _ = distillation.protocols_for_extension("howard_vala", {}, 3)
    assert any("triorthogonal" in p.protocol_name.lower() for p in hits)
    assert all(p.qudit_dim == 3 for p in hits)


def test_protocols_for_programmable_angle_includes_duclos_cianci() -> None:
    """A non-π/4 qubit angle should surface the programmable-MSD protocols
    and flag the Anderson-Jochym-O'Connor exclusion in the note."""
    hits, note = distillation.protocols_for_extension(
        "angle", {"theta": 2 * math.pi / 9}, 2
    )
    names = [p.protocol_name for p in hits]
    assert any("Duclos" in n for n in names), names
    assert any("Campbell" in n and "O'Gorman" in n for n in names), names
    assert all(p.qudit_dim == 2 for p in hits)
    assert "Anderson-Jochym-O'Connor" in note


def test_protocols_for_rnd_returns_research_note_with_ajoc_flag() -> None:
    """Haar-random extensions are not exactly synthesisable; should return
    empty protocol list AND surface the AJOC exclusion note at d=2."""
    hits, note = distillation.protocols_for_extension("rnd", {}, 2)
    assert hits == []
    assert "Anderson-Jochym-O'Connor" in note
    assert "Research may be needed" in note or "No catalog" in note


def test_ajoc_excluded_qubit() -> None:
    """AJOC helper: π/4 not excluded; non-π/4 angles are."""
    ex, reason = distillation.ajoc_excluded("angle", {"theta": math.pi / 4}, 2)
    assert ex is False
    assert "Clifford hierarchy" in reason

    ex, reason = distillation.ajoc_excluded("angle", {"theta": 2 * math.pi / 9}, 2)
    assert ex is True
    assert "Anderson-Jochym-O'Connor" in reason or "Clifford-hierarchy" in reason

    ex, _ = distillation.ajoc_excluded("rnd", {}, 2)
    assert ex is True


def test_ajoc_not_applicable_at_qudit_dim_3() -> None:
    """No AJOC-analogue for d≥3; helper returns False with an explanatory note."""
    ex, reason = distillation.ajoc_excluded("angle", {"theta": 2 * math.pi / 9}, 3)
    assert ex is False
    assert "d≥3" in reason or "d=3" in reason or "classification" in reason
