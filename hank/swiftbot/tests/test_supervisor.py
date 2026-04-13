"""Tests for swiftbot.supervisor + swiftbot.llm.

We never hit the real Anthropic API; `ScriptedLLM` returns pre-registered
Pydantic models in FIFO order.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from swiftbot.kb.cache import Cache
from swiftbot.llm import ScriptedLLM
from swiftbot.state import SawickiRecord
from swiftbot.supervisor import (
    ExtensionProposal,
    ExtensionSpec,
    GroupPriority,
    GroupSelection,
    run,
    stage1_explore_groups,
    stage2_propose_extensions,
)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def test_stage1_computes_sawicki_for_all_registered_d2_groups(tmp_path: Path) -> None:
    """Calling stage1 for d=2 should compute and cache Sawicki records for all
    5 registered d=2 groups (BI, BO, BT, clifford, hurwitz)."""
    selection_response = GroupSelection(
        selections=[
            GroupPriority(name="BI",       rationale="largest irreducible d=2 option"),
            GroupPriority(name="clifford", rationale="paper's canonical baseline"),
        ]
    )
    llm = ScriptedLLM([selection_response])
    with Cache(tmp_path / "cache.db") as cache:
        selection, verdicts = stage1_explore_groups(dim=2, cache=cache, llm=llm)
        # All 5 d=2 groups should have Sawicki records after stage 1.
        assert set(verdicts) == {"BI", "BO", "BT", "clifford", "hurwitz"}
        for name, rec in verdicts.items():
            assert rec.irreducible is True         # every d=2 registered group is irr
            assert cache.get_sawicki(rec.target_key) is not None
        # LLM only asked once — for the ranking.
        assert len(llm.history) == 1
        assert llm.history[0]["output_model"] == "GroupSelection"
    assert selection.selections[0].name == "BI"


def test_stage1_is_idempotent_via_cache(tmp_path: Path) -> None:
    """Calling stage1 twice must not re-run Sawicki on cached groups."""
    selection_response = GroupSelection(
        selections=[GroupPriority(name="BT", rationale="small test case")]
    )
    llm_a = ScriptedLLM([selection_response])
    llm_b = ScriptedLLM([selection_response])
    db = tmp_path / "cache.db"
    with Cache(db) as cache:
        stage1_explore_groups(dim=2, cache=cache, llm=llm_a)
        count_after_first = cache.count("sawicki_results")
    with Cache(db) as cache:
        stage1_explore_groups(dim=2, cache=cache, llm=llm_b)
        count_after_second = cache.count("sawicki_results")
    # No new Sawicki rows should have been written on the second run.
    assert count_after_first == count_after_second == 5


def test_stage1_rejects_unregistered_pick(tmp_path: Path) -> None:
    """If the LLM hallucinates a group name, we fail loudly rather than
    silently passing a bogus name forward."""
    selection_response = GroupSelection(
        selections=[GroupPriority(name="fake_group", rationale="not a thing")]
    )
    llm = ScriptedLLM([selection_response])
    with Cache(tmp_path / "cache.db") as cache:
        with pytest.raises(ValueError, match="unregistered group"):
            stage1_explore_groups(dim=2, cache=cache, llm=llm)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def test_stage2_returns_proposal(tmp_path: Path) -> None:
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="angle", params={"theta": 0.2}, rationale="sweep"),
        ExtensionSpec(kind="rnd",   params={},             rationale="control"),
    ])
    llm = ScriptedLLM([proposal])
    with Cache(tmp_path / "cache.db") as cache:
        # Minimal setup: put a group and its Sawicki so stage2 has context.
        import numpy as np
        from swiftbot.tools import groups as gmod
        mats = np.asarray(gmod.get_group("BI"))
        key = cache.put_group(mats, name="BI", source="registry:BI")
        verdict = SawickiRecord(
            target_key=key, verdict="inconclusive", commutant_dim=1,
            irreducible=True, min_distance_to_center=1.7,
            has_near_center_element=False, notes="",
        )
        cache.put_sawicki(verdict)
        out = stage2_propose_extensions("BI", verdict, cache=cache, llm=llm)
    assert len(out.extensions) == 2
    assert out.extensions[0].kind == "angle"
    assert out.extensions[1].kind == "rnd"


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

def test_run_d2_end_to_end_with_scripted_llm(tmp_path: Path) -> None:
    """One full invocation through the supervisor with scripted LLM responses."""
    selection = GroupSelection(selections=[
        GroupPriority(name="BI",       rationale="largest d=2 option"),
        GroupPriority(name="clifford", rationale="paper baseline"),
    ])
    proposal_bi = ExtensionProposal(extensions=[
        ExtensionSpec(kind="rnd", params={}, rationale="Haar baseline for BI"),
    ])
    proposal_cliff = ExtensionProposal(extensions=[
        ExtensionSpec(kind="angle", params={"theta": 0.7854}, rationale="paper P(π/4) baseline"),
        ExtensionSpec(kind="mat",   params={"source": "conj_P3pi4"}, rationale="paper best non-diagonal"),
    ])
    llm = ScriptedLLM([selection, proposal_bi, proposal_cliff])
    with Cache(tmp_path / "cache.db", run_id="test") as cache:
        result = run(dim=2, cache=cache, llm=llm, top_n=2)
    assert result.dim == 2
    assert len(result.selection.selections) == 2
    assert set(result.proposals) == {"BI", "clifford"}
    assert len(result.proposals["clifford"].extensions) == 2
    # Every registered d=2 group has a cached Sawicki record after the run.
    assert set(result.verdicts) == {"BI", "BO", "BT", "clifford", "hurwitz"}


def test_run_stops_when_llm_underfills(tmp_path: Path) -> None:
    """If the LLM picks fewer than top_n groups, run() cleanly uses what it got."""
    selection = GroupSelection(selections=[
        GroupPriority(name="hurwitz", rationale="smallest test case"),
    ])
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="rnd", params={}, rationale="baseline"),
    ])
    llm = ScriptedLLM([selection, proposal])
    with Cache(tmp_path / "cache.db") as cache:
        result = run(dim=2, cache=cache, llm=llm, top_n=5)
    assert len(result.proposals) == 1
    assert "hurwitz" in result.proposals
