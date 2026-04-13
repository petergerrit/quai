"""SQLite-backed cache for SWIFTbot pipeline artefacts.

One DB file (default `swiftbot/kb/cache.db`) with four tables:
    * `groups`          — finite matrix groups (blob-stored np.ndarray)
    * `sawicki_results` — universality verdicts keyed by target group
    * `qt_results`      — δ, Q_T measurements per (target, t, sample_id)
    * `provenance`      — when/where each row was written

Content-addressable keys: a group's key is SHA-256 over its *rounded,
lexicographically sorted* matrix contents. Two identical closures
therefore hash to the same key regardless of insertion order, which is
what we want for memoisation across sessions and machines.

Concurrency: SQLite with WAL mode allows concurrent readers + one writer.
For this pipeline we have a single-process agent, so no extra locking is
needed. Readers (tests, the LLM supervisor dashboard) are safe at any time.
"""
from __future__ import annotations

import hashlib
import io
import json
import socket
import sqlite3
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from swiftbot.state import (
    CoverageRecord,
    GroupRecord,
    PerTargetCoverage,
    Provenance,
    QTRecord,
    SCHEMA_VERSION,
    SawickiRecord,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "cache.db"


# ---------------------------------------------------------------------------
# Provenance + content hashing
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_sha() -> str | None:
    """Current HEAD SHA if we're inside a clean git tree, else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def current_provenance(run_id: str | None = None) -> Provenance:
    return Provenance(
        created_at=_utc_now_iso(),
        machine=socket.gethostname(),
        git_sha=_git_sha(),
        run_id=run_id,
    )


def matrices_key(matrices: np.ndarray, decimals: int = 5) -> str:
    """SHA-256 of the sorted rounded matrix list. Insertion-order-independent."""
    arr = np.asarray(matrices, dtype=complex)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D (n, d, d) array; got shape {arr.shape}")
    # Round real and imag independently; avoids -0.0 vs +0.0 hash divergence.
    re = np.round(arr.real, decimals)
    im = np.round(arr.imag, decimals)
    re[re == 0.0] = 0.0
    im[im == 0.0] = 0.0
    combined = np.concatenate([re, im], axis=-1)   # (n, d, 2d) — unique per matrix
    flat = combined.reshape(arr.shape[0], -1)
    # Sort by canonical byte representation so the list order doesn't matter.
    order = np.argsort([row.tobytes() for row in flat])
    h = hashlib.sha256()
    h.update(f"swiftbot-v{SCHEMA_VERSION}|d={arr.shape[1]}|n={arr.shape[0]}|dec={decimals}|".encode())
    h.update(flat[order].tobytes())
    return h.hexdigest()


def matrices_to_blob(matrices: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(matrices, dtype=complex), allow_pickle=False)
    return buf.getvalue()


def blob_to_matrices(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS groups (
    group_key      TEXT PRIMARY KEY,
    name           TEXT,
    d              INTEGER NOT NULL,
    size           INTEGER NOT NULL,
    source         TEXT NOT NULL DEFAULT 'unknown',
    projective     INTEGER NOT NULL DEFAULT 0,
    matrices_blob  BLOB    NOT NULL,
    created_at     TEXT    NOT NULL,
    machine        TEXT    NOT NULL,
    git_sha        TEXT,
    run_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_groups_name ON groups(name);
CREATE INDEX IF NOT EXISTS idx_groups_d    ON groups(d);

CREATE TABLE IF NOT EXISTS sawicki_results (
    target_key              TEXT PRIMARY KEY,
    verdict                 TEXT NOT NULL,
    commutant_dim           INTEGER NOT NULL,
    irreducible             INTEGER NOT NULL,
    min_distance_to_center  REAL NOT NULL,
    has_near_center_element INTEGER NOT NULL,
    notes                   TEXT,
    created_at              TEXT NOT NULL,
    machine                 TEXT NOT NULL,
    git_sha                 TEXT,
    run_id                  TEXT
);

CREATE TABLE IF NOT EXISTS qt_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_key      TEXT    NOT NULL,
    t               INTEGER NOT NULL,
    sample_id       INTEGER NOT NULL DEFAULT 0,
    ext_fingerprint TEXT    NOT NULL DEFAULT '',
    delta           REAL    NOT NULL,
    qt              REAL,
    q_opt           REAL,
    source_file     TEXT,
    created_at      TEXT    NOT NULL,
    machine         TEXT    NOT NULL,
    git_sha         TEXT,
    run_id          TEXT,
    UNIQUE(target_key, t, sample_id, ext_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_qt_target ON qt_results(target_key);

CREATE TABLE IF NOT EXISTS coverage_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    base_group_key     TEXT    NOT NULL,
    ext_fingerprint    TEXT    NOT NULL DEFAULT '',
    target_family_name TEXT    NOT NULL,
    max_depth          INTEGER NOT NULL,
    n_targets          INTEGER NOT NULL,
    visited            INTEGER NOT NULL,
    mean_dist          REAL    NOT NULL,
    max_dist           REAL    NOT NULL,
    hits_count         INTEGER NOT NULL,
    mean_t_count_hits  REAL,                -- schema v4: mean T-count over hit targets
    cost_method        TEXT    NOT NULL DEFAULT 'bfs_estimate',  -- schema v4
    certified          INTEGER NOT NULL DEFAULT 0,               -- schema v4
    cost_method_notes  TEXT    NOT NULL DEFAULT '',              -- schema v4
    per_target_json    TEXT    NOT NULL,
    created_at         TEXT    NOT NULL,
    machine            TEXT    NOT NULL,
    git_sha            TEXT,
    run_id             TEXT,
    UNIQUE(base_group_key, ext_fingerprint, target_family_name, max_depth)
);

CREATE INDEX IF NOT EXISTS idx_coverage_family  ON coverage_results(target_family_name);
CREATE INDEX IF NOT EXISTS idx_coverage_base    ON coverage_results(base_group_key);
"""


# ---------------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------------

class Cache:
    """Thin wrapper around a sqlite3 connection with typed accessors.

    Use as a context manager for auto-close::

        with Cache() as kb:
            key = kb.put_group(matrices, name="BI", source="registry:BI")
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets us share this connection across threads
        # (needed for supervisor.sweep with workers>1). We serialise actual
        # operations with self._lock below; WAL mode additionally serialises
        # writes at the SQLite level across any number of connections.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._lock = threading.Lock()
        self._init_schema()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # -- schema ---------------------------------------------------------------

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA_SQL)
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("version", str(SCHEMA_VERSION)),
            )

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()
        return int(row["value"]) if row else -1

    # -- groups ---------------------------------------------------------------

    def put_group(
        self,
        matrices: np.ndarray,
        *,
        name: str | None = None,
        source: str = "unknown",
        projective: bool = False,
        decimals: int = 5,
    ) -> str:
        """Store matrices (idempotent — returns existing key if already present)."""
        arr = np.asarray(matrices, dtype=complex)
        if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
            raise ValueError(f"matrices must have shape (n, d, d); got {arr.shape}")
        key = matrices_key(arr, decimals=decimals)
        prov = current_provenance(run_id=self.run_id)
        blob = matrices_to_blob(arr)
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR IGNORE INTO groups (
                    group_key, name, d, size, source, projective, matrices_blob,
                    created_at, machine, git_sha, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    key, name, arr.shape[1], arr.shape[0], source, int(projective),
                    blob, prov.created_at, prov.machine, prov.git_sha, prov.run_id,
                ),
            )
        return key

    def get_group(self, group_key: str) -> tuple[GroupRecord, np.ndarray] | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT group_key, name, d, size, source, projective, matrices_blob
                   FROM groups WHERE group_key = ?""",
                (group_key,),
            ).fetchone()
        if row is None:
            return None
        record = GroupRecord(
            group_key=row["group_key"],
            name=row["name"],
            d=row["d"],
            size=row["size"],
            source=row["source"],
            projective=bool(row["projective"]),
        )
        return record, blob_to_matrices(row["matrices_blob"])

    def list_groups(self, d: int | None = None) -> list[GroupRecord]:
        sql = "SELECT group_key, name, d, size, source, projective FROM groups"
        args: tuple = ()
        if d is not None:
            sql += " WHERE d = ?"
            args = (d,)
        sql += " ORDER BY d, size, name"
        with self._lock:
            rows = list(self.conn.execute(sql, args))
        return [
            GroupRecord(
                group_key=row["group_key"], name=row["name"], d=row["d"],
                size=row["size"], source=row["source"],
                projective=bool(row["projective"]),
            )
            for row in rows
        ]

    # -- sawicki --------------------------------------------------------------

    def put_sawicki(self, record: SawickiRecord) -> None:
        prov = current_provenance(run_id=self.run_id)
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO sawicki_results (
                    target_key, verdict, commutant_dim, irreducible,
                    min_distance_to_center, has_near_center_element, notes,
                    created_at, machine, git_sha, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.target_key, record.verdict, record.commutant_dim,
                    int(record.irreducible), record.min_distance_to_center,
                    int(record.has_near_center_element), record.notes,
                    prov.created_at, prov.machine, prov.git_sha, prov.run_id,
                ),
            )

    def get_sawicki(self, target_key: str) -> SawickiRecord | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT target_key, verdict, commutant_dim, irreducible,
                          min_distance_to_center, has_near_center_element, notes
                   FROM sawicki_results WHERE target_key = ?""",
                (target_key,),
            ).fetchone()
        if row is None:
            return None
        return SawickiRecord(
            target_key=row["target_key"],
            verdict=row["verdict"],
            commutant_dim=row["commutant_dim"],
            irreducible=bool(row["irreducible"]),
            min_distance_to_center=row["min_distance_to_center"],
            has_near_center_element=bool(row["has_near_center_element"]),
            notes=row["notes"] or "",
        )

    # -- qt -------------------------------------------------------------------

    def put_qt(self, record: QTRecord) -> None:
        prov = current_provenance(run_id=self.run_id)
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO qt_results (
                    target_key, t, sample_id, ext_fingerprint,
                    delta, qt, q_opt, source_file,
                    created_at, machine, git_sha, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.target_key, record.t, record.sample_id, record.ext_fingerprint,
                    record.delta, record.qt, record.q_opt, record.source_file,
                    prov.created_at, prov.machine, prov.git_sha, prov.run_id,
                ),
            )

    def get_qt(
        self,
        target_key: str,
        t: int,
        sample_id: int = 0,
        *,
        ext_fingerprint: str = "",
    ) -> QTRecord | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT target_key, t, sample_id, ext_fingerprint,
                          delta, qt, q_opt, source_file
                   FROM qt_results
                   WHERE target_key = ? AND t = ? AND sample_id = ? AND ext_fingerprint = ?""",
                (target_key, t, sample_id, ext_fingerprint),
            ).fetchone()
        if row is None:
            return None
        return QTRecord(**{k: row[k] for k in row.keys()})

    def list_qt(self, target_key: str) -> list[QTRecord]:
        with self._lock:
            rows = list(self.conn.execute(
                """SELECT target_key, t, sample_id, ext_fingerprint,
                          delta, qt, q_opt, source_file
                   FROM qt_results WHERE target_key = ?
                   ORDER BY t, sample_id, ext_fingerprint""",
                (target_key,),
            ))
        return [
            QTRecord(
                target_key=row["target_key"], t=row["t"],
                sample_id=row["sample_id"],
                ext_fingerprint=row["ext_fingerprint"] or "",
                delta=row["delta"],
                qt=row["qt"], q_opt=row["q_opt"],
                source_file=row["source_file"],
            )
            for row in rows
        ]

    # -- coverage -------------------------------------------------------------

    def put_coverage(self, record: CoverageRecord) -> None:
        prov = current_provenance(run_id=self.run_id)
        per_target_json = json.dumps(
            [pt.model_dump() for pt in record.per_target], separators=(",", ":"),
        )
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO coverage_results (
                    base_group_key, ext_fingerprint, target_family_name, max_depth,
                    n_targets, visited, mean_dist, max_dist, hits_count,
                    mean_t_count_hits, cost_method, certified, cost_method_notes,
                    per_target_json, created_at, machine, git_sha, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.base_group_key, record.ext_fingerprint,
                    record.target_family_name, record.max_depth,
                    record.n_targets, record.visited,
                    record.mean_dist, record.max_dist, record.hits_count,
                    record.mean_t_count_hits,
                    record.cost_method, int(record.certified), record.cost_method_notes,
                    per_target_json,
                    prov.created_at, prov.machine, prov.git_sha, prov.run_id,
                ),
            )

    def _row_to_coverage(self, row) -> CoverageRecord:
        per_target = [
            PerTargetCoverage.model_validate(pt)
            for pt in json.loads(row["per_target_json"])
        ]
        return CoverageRecord(
            base_group_key=row["base_group_key"],
            ext_fingerprint=row["ext_fingerprint"],
            target_family_name=row["target_family_name"],
            max_depth=row["max_depth"],
            n_targets=row["n_targets"],
            visited=row["visited"],
            mean_dist=row["mean_dist"],
            max_dist=row["max_dist"],
            hits_count=row["hits_count"],
            mean_t_count_hits=row["mean_t_count_hits"],
            cost_method=row["cost_method"],
            certified=bool(row["certified"]),
            cost_method_notes=row["cost_method_notes"] or "",
            per_target=per_target,
        )

    def get_coverage(
        self,
        base_group_key: str,
        ext_fingerprint: str,
        target_family_name: str,
        max_depth: int,
    ) -> CoverageRecord | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT * FROM coverage_results
                   WHERE base_group_key = ? AND ext_fingerprint = ?
                     AND target_family_name = ? AND max_depth = ?""",
                (base_group_key, ext_fingerprint, target_family_name, max_depth),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_coverage(row)

    def list_coverage(
        self,
        target_family_name: str | None = None,
        base_group_key: str | None = None,
    ) -> list[CoverageRecord]:
        sql = "SELECT * FROM coverage_results"
        clauses: list[str] = []
        args: list = []
        if target_family_name is not None:
            clauses.append("target_family_name = ?")
            args.append(target_family_name)
        if base_group_key is not None:
            clauses.append("base_group_key = ?")
            args.append(base_group_key)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY mean_dist ASC"
        with self._lock:
            rows = list(self.conn.execute(sql, tuple(args)))
        return [self._row_to_coverage(row) for row in rows]

    # -- misc -----------------------------------------------------------------

    def count(self, table: str) -> int:
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()
        return int(row["n"])
