"""Tests for swiftbot.kb.cache — SQLite knowledge base.

Focus: content-addressable keys are deterministic, all record types
round-trip losslessly, schema init is idempotent, list queries filter.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from swiftbot.kb import cache as cachemod
from swiftbot.kb.cache import (
    Cache,
    blob_to_matrices,
    matrices_key,
    matrices_to_blob,
)
from swiftbot.state import (
    SCHEMA_VERSION,
    GroupRecord,
    QTRecord,
    SawickiRecord,
)
from swiftbot.tools import groups as gmod


# ---------------------------------------------------------------------------
# Content-addressable keys
# ---------------------------------------------------------------------------

def test_matrices_key_is_deterministic() -> None:
    bi = np.asarray(gmod.get_group("BI"))
    assert matrices_key(bi) == matrices_key(bi)


def test_matrices_key_is_insertion_order_independent() -> None:
    bi = np.asarray(gmod.get_group("BI"))
    shuffled = bi[np.random.default_rng(0).permutation(bi.shape[0])]
    assert matrices_key(bi) == matrices_key(shuffled)


def test_matrices_key_distinguishes_groups() -> None:
    bi = np.asarray(gmod.get_group("BI"))
    bo = np.asarray(gmod.get_group("BO"))
    assert matrices_key(bi) != matrices_key(bo)


def test_matrices_blob_roundtrip_is_exact() -> None:
    bi = np.asarray(gmod.get_group("BI"))
    blob = matrices_to_blob(bi)
    reloaded = blob_to_matrices(blob)
    assert np.array_equal(bi, reloaded)


# ---------------------------------------------------------------------------
# Schema + lifecycle
# ---------------------------------------------------------------------------

def test_fresh_cache_is_empty_and_has_schema(tmp_path: Path) -> None:
    with Cache(tmp_path / "cache.db") as kb:
        assert kb.schema_version == SCHEMA_VERSION
        assert kb.count("groups") == 0
        assert kb.count("sawicki_results") == 0
        assert kb.count("qt_results") == 0


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "cache.db"
    with Cache(path):
        pass
    # Re-opening must not error or duplicate rows.
    with Cache(path) as kb:
        assert kb.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Groups table
# ---------------------------------------------------------------------------

def test_put_and_get_group_roundtrip(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db") as kb:
        key = kb.put_group(bi, name="BI", source="registry:BI")
        record, loaded = kb.get_group(key)

    assert record.name == "BI"
    assert record.d == 2
    assert record.size == 120
    assert record.source == "registry:BI"
    assert np.array_equal(loaded, bi)


def test_put_group_is_idempotent(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db") as kb:
        k1 = kb.put_group(bi, name="BI")
        k2 = kb.put_group(bi, name="BI")
        assert k1 == k2
        assert kb.count("groups") == 1


def test_list_groups_filters_by_dim(tmp_path: Path) -> None:
    with Cache(tmp_path / "cache.db") as kb:
        kb.put_group(np.asarray(gmod.get_group("BI")), name="BI")
        kb.put_group(np.asarray(gmod.get_group("S108")), name="S108")
        kb.put_group(np.asarray(gmod.get_group("s60")), name="s60")
        assert len(kb.list_groups()) == 3
        assert len(kb.list_groups(d=2)) == 1
        assert len(kb.list_groups(d=3)) == 1
        assert len(kb.list_groups(d=4)) == 1


def test_get_missing_group_returns_none(tmp_path: Path) -> None:
    with Cache(tmp_path / "cache.db") as kb:
        assert kb.get_group("0" * 64) is None


# ---------------------------------------------------------------------------
# Sawicki table
# ---------------------------------------------------------------------------

def test_put_and_get_sawicki(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db") as kb:
        key = kb.put_group(bi, name="BI")
        rec = SawickiRecord(
            target_key=key,
            verdict="inconclusive",
            commutant_dim=1,
            irreducible=True,
            min_distance_to_center=0.9,
            has_near_center_element=False,
            notes="finite group: inconclusive by Sawicki sufficient clause",
        )
        kb.put_sawicki(rec)
        out = kb.get_sawicki(key)

    assert out == rec


def test_put_sawicki_upserts(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db") as kb:
        key = kb.put_group(bi, name="BI")
        kb.put_sawicki(SawickiRecord(
            target_key=key, verdict="inconclusive",
            commutant_dim=1, irreducible=True,
            min_distance_to_center=0.9, has_near_center_element=False,
        ))
        # Subsequent put with different verdict must replace.
        kb.put_sawicki(SawickiRecord(
            target_key=key, verdict="universal",
            commutant_dim=1, irreducible=True,
            min_distance_to_center=0.1, has_near_center_element=True,
        ))
        out = kb.get_sawicki(key)
        assert out is not None and out.verdict == "universal"
        assert kb.count("sawicki_results") == 1


# ---------------------------------------------------------------------------
# QT table
# ---------------------------------------------------------------------------

def test_put_and_list_qt(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db") as kb:
        key = kb.put_group(bi, name="BI")
        for t in (5, 50, 500):
            kb.put_qt(QTRecord(
                target_key=key, t=t, sample_id=0,
                delta=0.9, qt=math.log(120) / math.log(1 / 0.9),
                q_opt=2.0,
            ))
        rows = kb.list_qt(key)
        assert [r.t for r in rows] == [5, 50, 500]
        assert all(r.target_key == key for r in rows)


def test_put_qt_unique_per_target_t_sample(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db") as kb:
        key = kb.put_group(bi, name="BI")
        kb.put_qt(QTRecord(target_key=key, t=50, delta=0.9, qt=1.0))
        # Re-put for same (target, t, sample_id) should update, not duplicate.
        kb.put_qt(QTRecord(target_key=key, t=50, delta=0.85, qt=1.5))
        assert kb.count("qt_results") == 1
        row = kb.get_qt(key, t=50)
        assert row is not None and row.delta == pytest.approx(0.85)


def test_get_missing_qt_returns_none(tmp_path: Path) -> None:
    with Cache(tmp_path / "cache.db") as kb:
        assert kb.get_qt("0" * 64, t=5) is None


# ---------------------------------------------------------------------------
# Provenance sanity
# ---------------------------------------------------------------------------

def test_concurrent_writes_from_multiple_threads(tmp_path: Path) -> None:
    """Regression: supervisor.sweep(workers>1) shares a Cache across threads.
    Before the lock + check_same_thread=False fix, this raised ProgrammingError
    ('SQLite objects created in a thread can only be used in that same thread')."""
    import threading

    from swiftbot.state import QTRecord, SawickiRecord

    with Cache(tmp_path / "cache.db") as kb:
        bi_key = kb.put_group(
            np.asarray(gmod.get_group("BI")), name="BI", source="registry:BI",
        )

        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def writer(i: int) -> None:
            try:
                kb.put_qt(QTRecord(
                    target_key=bi_key, t=5, sample_id=i,
                    delta=0.5, qt=2.0, q_opt=2.0,
                    ext_fingerprint=f"fp-{i}",
                ))
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent writes errored: {errors[:3]}"
        assert kb.count("qt_results") == 20

        # Readers can also run from a different thread.
        out: list[int] = []
        read_thread = threading.Thread(
            target=lambda: out.append(len(kb.list_qt(bi_key)))
        )
        read_thread.start()
        read_thread.join()
        assert out == [20]


def test_rows_carry_provenance(tmp_path: Path) -> None:
    bi = np.asarray(gmod.get_group("BI"))
    with Cache(tmp_path / "cache.db", run_id="test-run") as kb:
        key = kb.put_group(bi, name="BI")
        raw = kb.conn.execute(
            "SELECT created_at, machine, run_id FROM groups WHERE group_key = ?",
            (key,),
        ).fetchone()
    assert raw["created_at"].endswith("Z")
    assert raw["machine"]  # hostname string
    assert raw["run_id"] == "test-run"
