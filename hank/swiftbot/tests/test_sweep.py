"""Tests for supervisor.sweep + CLI `sweep` command.

Uses ScriptedLLM (no real API calls) and the live qco subprocess with tiny
parameters (t=5, samples=1). Whole suite stays under a few seconds.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from swiftbot.kb.cache import Cache
from swiftbot.llm import ScriptedLLM
from swiftbot.stages import s3_efficiency as s3
from swiftbot.supervisor import (
    ExtensionProposal,
    ExtensionSpec,
    GroupPriority,
    GroupSelection,
    format_sweep_table,
    sweep,
)


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_sweep_evaluates_every_proposed_extension(tmp_path: Path) -> None:
    """Feed the supervisor a ranking + one proposal per group, confirm that
    sweep runs qco for each (group, extension) pair and records Q_T, and
    that Stage-4/5 catalogs decorate each evaluation."""
    selection = GroupSelection(selections=[
        GroupPriority(name="clifford", rationale="paper canonical base"),
    ])
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="rnd", params={}, rationale="Haar baseline"),
        ExtensionSpec(kind="angle", params={"theta": math.pi / 4},
                      rationale="paper's P(π/4) baseline"),
    ])
    llm = ScriptedLLM([selection, proposal])

    with Cache(tmp_path / "cache.db", run_id="sweep-test") as cache:
        result = sweep(
            dim=2,
            t=5,
            sample_size=1,
            cache=cache,
            llm=llm,
            top_n=1,
            timeout_s=60,
            verbose=False,
        )

    assert result.dim == 2
    assert result.t == 5
    assert len(result.evaluations) == 2
    assert all(e.group_name == "clifford" for e in result.evaluations)

    # Both must succeed and have at least one QTRecord.
    for ev in result.evaluations:
        assert ev.error is None, f"{ev.ext_kind}: {ev.error}"
        assert ev.qt_records, f"{ev.ext_kind}: no QTRecords"
        assert ev.best_qt is not None and math.isfinite(ev.best_qt)
        assert 0 < ev.best_delta < 1
        # Stage-4 catalog decoration:  'clifford' has curated codes.
        assert ev.codes_found, f"{ev.ext_kind}: expected at least one code for clifford"
        assert any("Steane" in c.name for c in ev.codes_found)

    # Stage-5: the angle=π/4 extension should map to the qubit-T distillation
    # family; the rnd extension should not (no concrete family tag).
    evmap = {ev.ext_kind: ev for ev in result.evaluations}
    assert evmap["angle"].distillation_found, "angle π/4 should have distillations"
    assert any(
        "Bravyi-Kitaev" in p.protocol_name
        for p in evmap["angle"].distillation_found
    )
    assert evmap["rnd"].distillation_found == []
    assert "Research may be needed" in evmap["rnd"].distillation_note or "no catalog" in evmap["rnd"].distillation_note.lower()

    # Cache: one QT row per (sample, ext_fingerprint) — 1×2 = 2 rows here.
    with Cache(tmp_path / "cache.db") as cache:
        assert cache.count("qt_results") >= 2


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_sweep_continues_past_failing_extension(tmp_path: Path) -> None:
    """If one extension raises (e.g. a `mat` with the wrong shape), the sweep
    should record the error and still evaluate the other proposals."""
    selection = GroupSelection(selections=[
        GroupPriority(name="clifford", rationale="paper baseline"),
    ])
    # Deliberately malformed: a 3×3 matrix for a d=2 base group.
    bad_mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="mat", params={"matrix": bad_mat},
                      rationale="intentionally wrong-shape to test error path"),
        ExtensionSpec(kind="rnd", params={}, rationale="Haar control"),
    ])
    llm = ScriptedLLM([selection, proposal])

    with Cache(tmp_path / "cache.db") as cache:
        result = sweep(
            dim=2, t=5, sample_size=1,
            cache=cache, llm=llm, top_n=1, timeout_s=60, verbose=False,
        )

    assert len(result.evaluations) == 2
    errs = [e for e in result.evaluations if e.error is not None]
    oks  = [e for e in result.evaluations if e.error is None]
    assert len(errs) == 1 and errs[0].ext_kind == "mat"
    assert len(oks) == 1 and oks[0].ext_kind == "rnd"


def test_sweep_respects_max_extensions_per_group(tmp_path: Path) -> None:
    """If max_extensions_per_group=1 we should only evaluate the first proposal."""
    selection = GroupSelection(selections=[
        GroupPriority(name="hurwitz", rationale="tiny"),
    ])
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="rnd", params={}, rationale="A"),
        ExtensionSpec(kind="rnd", params={}, rationale="B"),
        ExtensionSpec(kind="rnd", params={}, rationale="C"),
    ])
    llm = ScriptedLLM([selection, proposal])

    if not s3.MAIN_PY.exists():
        pytest.skip("qco-main_opt not available")

    with Cache(tmp_path / "cache.db") as cache:
        result = sweep(
            dim=2, t=5, sample_size=1,
            cache=cache, llm=llm, top_n=1, max_extensions_per_group=1,
            timeout_s=60, verbose=False,
        )
    assert len(result.evaluations) == 1


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_sweep_with_coverage_attaches_coverage_records(tmp_path: Path) -> None:
    """include_coverage=True populates SweepEvaluation.coverage_records with
    one CoverageRecord per (base × registered-family-at-matching-dim) pair."""
    selection = GroupSelection(selections=[
        GroupPriority(name="clifford", rationale="d=2 baseline for coverage"),
    ])
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="angle", params={"theta": 2 * math.pi / 9},
                      rationale="P(2π/9) should exact-cover 9th-root targets"),
    ])
    llm = ScriptedLLM([selection, proposal])
    with Cache(tmp_path / "cache.db", run_id="coverage-integ") as cache:
        result = sweep(
            dim=2, t=5, sample_size=1,
            cache=cache, llm=llm, top_n=1,
            timeout_s=60, verbose=False,
            include_coverage=True,
            coverage_bases=("clifford",),
            coverage_max_depth=6,
            coverage_n_parametric=2,
        )
    assert len(result.evaluations) == 1
    ev = result.evaluations[0]
    assert ev.error is None
    # Two target families registered at d=2: lamm_sigma36 and lamm_sigma72.
    # With one coverage base (clifford), we expect 2 records.
    assert len(ev.coverage_records) == 2
    names = {r.target_family_name for r in ev.coverage_records}
    assert names == {"lamm_sigma36", "lamm_sigma72"}
    # The 9th-root extension should hit every discrete target exactly:
    for rec in ev.coverage_records:
        disc_distances = [
            pt.distance for pt in rec.per_target
            if pt.label.startswith("R_Z(2π·")
        ]
        assert all(d < 1e-6 for d in disc_distances), (
            f"{rec.target_family_name}: discrete targets not exact"
        )


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_sweep_workers_produces_same_evaluation_set(tmp_path: Path) -> None:
    """workers=1 and workers>1 must yield the same set of SweepEvaluations
    (identical ordering, because _worker places results at their stable index)."""
    selection = GroupSelection(selections=[
        GroupPriority(name="hurwitz", rationale="tiny d=2 base for fast parallel test"),
    ])
    proposal = ExtensionProposal(extensions=[
        ExtensionSpec(kind="rnd", params={}, rationale="A"),
        ExtensionSpec(kind="angle", params={"theta": math.pi / 4}, rationale="B"),
        ExtensionSpec(kind="rnd", params={}, rationale="C"),
    ])

    def _run(workers: int, db_name: str) -> list[tuple[str, str]]:
        with Cache(tmp_path / db_name) as cache:
            result = sweep(
                dim=2, t=5, sample_size=1,
                cache=cache, llm=ScriptedLLM([selection, proposal]),
                top_n=1, timeout_s=60, verbose=False,
                workers=workers,
            )
        return [(e.group_name, e.ext_kind) for e in result.evaluations]

    serial   = _run(1, "serial.db")
    parallel = _run(3, "parallel.db")

    # Same (group, ext_kind) sequence — ordering is preserved by the stable index.
    assert serial == parallel
    assert len(serial) == 3
    assert [p[1] for p in serial] == ["rnd", "angle", "rnd"]


def test_format_sweep_table_handles_mixed_results() -> None:
    """format_sweep_table must gracefully handle both successful and failed
    evaluations, and sort by best_qt ascending (best first)."""
    from swiftbot.supervisor import SweepEvaluation, SweepResult
    evals = [
        SweepEvaluation(group_name="C1", ext_kind="rnd",
                        best_qt=5.0, best_delta=0.5),
        SweepEvaluation(group_name="C2", ext_kind="angle",
                        best_qt=2.0, best_delta=0.3),
        SweepEvaluation(group_name="C3", ext_kind="howard_vala",
                        error="NotImplementedError: soon"),
    ]
    result = SweepResult(
        dim=2, t=5, sample_size=1,
        selection=GroupSelection(selections=[
            GroupPriority(name="C1", rationale="a"),
        ]),
        verdicts={},
        proposals={},
        evaluations=evals,
    )
    table = format_sweep_table(result)
    # Best (lowest Q_T) should appear first.
    c2_line = next(ln for ln in table.splitlines() if ln.startswith("C2"))
    c1_line = next(ln for ln in table.splitlines() if ln.startswith("C1"))
    c3_line = next(ln for ln in table.splitlines() if ln.startswith("C3"))
    idx_c2 = table.index(c2_line)
    idx_c1 = table.index(c1_line)
    idx_c3 = table.index(c3_line)
    assert idx_c2 < idx_c1 < idx_c3                 # failures sort last
    assert "ERR" in c3_line
