"""Tests for swiftbot.targets + swiftbot.stages.target_coverage + cache hooks.

Covers:
  * TargetFamily materialisation (discrete + parametric)
  * Registered Lamm families (lamm_sigma36 / lamm_sigma72)
  * BFS coverage kernel matches Scope-A numbers on (Clifford, P(2π/9))
  * High-level evaluate_coverage_by_name end-to-end (no cache + with cache)
  * Cache round-trip of CoverageRecord
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from swiftbot.kb.cache import Cache, matrices_key
from swiftbot.stages.target_coverage import (
    DEFAULT_MAX_UNIQUE,
    evaluate_coverage,
    evaluate_coverage_by_name,
)
from swiftbot.state import CoverageRecord, PerTargetCoverage
from swiftbot.targets import get_target_family, list_target_families
from swiftbot.tools import groups as gmod


def _rz(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2), 0.0],
         [0.0, np.exp(1j * phi / 2)]],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# TargetFamily materialisation
# ---------------------------------------------------------------------------

def test_lamm_sigma36_registered_and_sized() -> None:
    fam = get_target_family("lamm_sigma36")
    assert fam.qudit_dim == 2
    assert len(fam.discrete) == 8
    mats = fam.materialize(n_parametric_samples=3, rng_seed=0)
    assert len(mats) == 8 + 3 * 5           # 8 discrete + 3 θ × 5 multipliers


def test_lamm_sigma72_larger_discrete() -> None:
    fam36 = get_target_family("lamm_sigma36")
    fam72 = get_target_family("lamm_sigma72")
    assert len(fam72.discrete) > len(fam36.discrete)
    # Σ(72×3) adds 18th-root phases that reduce mod gcd to non-9th-roots.
    new_labels = {l for l, _ in fam72.discrete} - {l for l, _ in fam36.discrete}
    assert new_labels, "Σ(72×3) should add at least one new discrete label"


def test_materialize_is_deterministic_under_seed() -> None:
    fam = get_target_family("lamm_sigma36")
    a = [(l, m.tobytes()) for l, m in fam.materialize(5, rng_seed=42)]
    b = [(l, m.tobytes()) for l, m in fam.materialize(5, rng_seed=42)]
    assert a == b


# ---------------------------------------------------------------------------
# Pure coverage kernel — paper reproduction
# ---------------------------------------------------------------------------

def test_kernel_reaches_all_9th_roots_exactly_with_P_2pi_9() -> None:
    """P(2π/9) via Clifford BFS must exactly reach all 8 R_Z(2π·k/9) targets."""
    gens = gmod.get_generating_set("clifford")
    T = _rz(2 * np.pi / 9)
    targets = [
        (f"R_Z(2π·{k}/9)", _rz(2 * np.pi * k / 9))
        for k in range(1, 9)
    ]
    k = evaluate_coverage(gens, T, targets, max_depth=6, eps_hit=1e-2)
    assert k["hits_count"] == 8, f"got {k['hits_count']}/8 hits"
    assert k["max_dist"] < 1e-6, (
        f"all 8 targets should be exact; max_dist = {k['max_dist']:.2e}"
    )


def test_kernel_canonical_T_misses_all_9th_roots() -> None:
    """Canonical Clifford+T (P(π/4)) cannot reach any non-trivial 9th-root phase."""
    gens = gmod.get_generating_set("clifford")
    T = _rz(np.pi / 4)
    targets = [
        (f"R_Z(2π·{k}/9)", _rz(2 * np.pi * k / 9))
        for k in range(1, 9)
    ]
    k = evaluate_coverage(gens, T, targets, max_depth=8, eps_hit=1e-2)
    assert k["hits_count"] == 0
    # Mean distance should be well above the ε threshold.
    assert k["mean_dist"] > 0.05


def test_kernel_tracks_t_count_per_path() -> None:
    """Each target hit via BFS records the T/T† count along its specific path.

    For Clifford + P(2π/9) the target R_Z(2π·3/9) should be hit by the word
    `T·T·T` (3 uses of T, no Clifford), giving t_count = 3 at depth 3.
    """
    gens = gmod.get_generating_set("clifford")
    T = _rz(2 * np.pi / 9)
    targets = [
        (f"R_Z(2π·{k}/9)", _rz(2 * np.pi * k / 9))
        for k in range(1, 9)
    ]
    k = evaluate_coverage(gens, T, targets, max_depth=8, eps_hit=1e-6)
    # Every discrete target is reached; check t_count is recorded for each.
    for tc in k["per_target_t_count_first_hit"]:
        assert tc is not None and tc >= 1
    # R_Z(2π·1/9) is T itself → t_count should be 1.
    assert k["per_target_t_count_first_hit"][0] == 1
    # R_Z(2π·2/9) is T·T → t_count = 2.
    assert k["per_target_t_count_first_hit"][1] == 2
    # R_Z(2π·k/9) for k=1..8 should have t_count ≤ k (at worst T^k or T†^(9-k))
    for k_idx, tc in enumerate(k["per_target_t_count_first_hit"], start=1):
        min_k = min(k_idx, 9 - k_idx)  # T^k or T†^(9-k), whichever shorter
        assert tc <= min_k, f"target k={k_idx}: t_count={tc} > {min_k}"


def test_record_has_method_label_and_not_certified() -> None:
    T = _rz(2 * np.pi / 9)
    rec = evaluate_coverage_by_name(
        "clifford", T, "lamm_sigma36",
        max_depth=5, n_parametric_samples=2,
    )
    assert rec.cost_method == "bfs_estimate"
    assert rec.certified is False
    assert "Ross-Selinger" in rec.cost_method_notes or "bfs_estimate" in rec.cost_method_notes.lower()
    # Mean t-count over hit targets is set (some targets were hit)
    assert rec.hits_count > 0
    assert rec.mean_t_count_hits is not None and rec.mean_t_count_hits > 0


def test_certified_method_catalog_knows_canonical_clifford_T() -> None:
    from swiftbot.stages.target_coverage import certified_method_for
    info = certified_method_for("clifford", "P(π/4)")
    assert info is not None
    method, citation = info
    assert method == "ross_selinger_exact"
    assert "Ross" in citation


def test_certified_method_catalog_unknown_pair() -> None:
    from swiftbot.stages.target_coverage import certified_method_for
    assert certified_method_for("clifford", "P(2π/9)") is None
    assert certified_method_for("hurwitz",  "P(π/4)") is None


def test_kernel_respects_max_unique() -> None:
    """max_unique caps the BFS so huge branching doesn't OOM."""
    gens = gmod.get_generating_set("clifford")
    T = _rz(np.sqrt(2))  # irrational angle → infinite closure
    targets = [("R_Z(0.3)", _rz(0.3))]
    k = evaluate_coverage(
        gens, T, targets, max_depth=100, max_unique=500,
    )
    assert k["visited"] <= 500


# ---------------------------------------------------------------------------
# High-level API + cache
# ---------------------------------------------------------------------------

def test_evaluate_by_name_without_cache() -> None:
    T = _rz(2 * np.pi / 9)
    rec = evaluate_coverage_by_name(
        "clifford", T, "lamm_sigma36",
        max_depth=6, n_parametric_samples=3,
    )
    assert isinstance(rec, CoverageRecord)
    assert rec.target_family_name == "lamm_sigma36"
    assert rec.n_targets == 8 + 3 * 5
    # All 8 discrete 9th-root targets must be reached exactly.
    discrete = [pt for pt in rec.per_target if pt.label.startswith("R_Z(2π·")]
    assert len(discrete) == 8
    assert all(pt.distance < 1e-6 for pt in discrete)


def test_evaluate_by_name_with_cache_round_trip(tmp_path: Path) -> None:
    T = _rz(2 * np.pi / 9)
    db = tmp_path / "cache.db"
    ext_fp = "test-ext-fp"
    with Cache(db) as cache:
        rec = evaluate_coverage_by_name(
            "clifford", T, "lamm_sigma36",
            max_depth=5, n_parametric_samples=2,
            cache=cache, ext_fingerprint=ext_fp,
        )
        # Group must have been registered as a side effect.
        assert cache.count("groups") == 1
        assert cache.count("coverage_results") == 1

        # Round-trip lookup
        loaded = cache.get_coverage(
            base_group_key=rec.base_group_key,
            ext_fingerprint=ext_fp,
            target_family_name="lamm_sigma36",
            max_depth=5,
        )
    assert loaded is not None
    assert loaded.mean_dist == pytest.approx(rec.mean_dist, abs=1e-12)
    assert loaded.hits_count == rec.hits_count
    assert len(loaded.per_target) == rec.n_targets


def test_list_coverage_filters(tmp_path: Path) -> None:
    with Cache(tmp_path / "cache.db") as cache:
        # Two extensions against same family
        T1 = _rz(2 * np.pi / 9)
        T2 = _rz(np.pi / 4)
        evaluate_coverage_by_name(
            "clifford", T1, "lamm_sigma36",
            max_depth=5, n_parametric_samples=2,
            cache=cache, ext_fingerprint="ext-A",
        )
        evaluate_coverage_by_name(
            "clifford", T2, "lamm_sigma36",
            max_depth=5, n_parametric_samples=2,
            cache=cache, ext_fingerprint="ext-B",
        )
        rows = cache.list_coverage(target_family_name="lamm_sigma36")
        assert len(rows) == 2
        # list_coverage sorts by mean_dist ASC, so the P(2π/9) row (A) is first.
        assert rows[0].ext_fingerprint == "ext-A"
        assert rows[0].mean_dist < rows[1].mean_dist
