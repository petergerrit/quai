"""Pydantic state models for SWIFTbot.

These are the structured records that flow through the pipeline stages and
that the SQLite cache persists. They deliberately do NOT carry the heavy
numerical artefacts (matrix lists) — those are stored as BLOBs in the
cache and loaded lazily. Each record references matrices by a
content-addressable `*_key` (SHA-256 hash of the matrix list).

Versioning: if you add/rename a field, bump the corresponding
`SCHEMA_VERSION` below so older DB files can be migrated or rebuilt.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 4


# ---------------------------------------------------------------------------
# Provenance — every inserted row carries one of these.
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    created_at: str                          # ISO-8601 UTC, e.g. 2026-04-12T19:30:00Z
    machine: str                             # socket.gethostname()
    git_sha: str | None = None               # None outside a git repo or if dirty
    run_id: str | None = None                # optional run label for grouping


# ---------------------------------------------------------------------------
# Stage-1 artefacts — finite matrix groups.
# ---------------------------------------------------------------------------

class GroupRecord(BaseModel):
    """Metadata for one finite matrix group. Actual matrices live in a blob
    keyed by `group_key` in the `groups` table."""

    group_key: str                           # content hash of the matrix list
    name: str | None = None                  # human-readable label ("BI", "S108", ...)
    d: int = Field(..., gt=0)
    size: int = Field(..., gt=0)             # |G| — number of matrices stored
    source: str = "unknown"                  # "registry:<name>" | "custom" | "extension:<parent_key>"
    projective: bool = False                 # closed in PU(d) rather than U(d)?


# ---------------------------------------------------------------------------
# Stage-2 artefacts — Sawicki universality check.
# ---------------------------------------------------------------------------

SawickiVerdict = Literal["universal", "not_universal", "inconclusive"]


class SawickiRecord(BaseModel):
    """Result of check_universality on a finite group (or generating set after
    closure). Keyed by the same `target_key` used in the groups / extensions
    tables so joins are trivial."""

    target_key: str
    verdict: SawickiVerdict
    commutant_dim: int
    irreducible: bool
    min_distance_to_center: float
    has_near_center_element: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Stage-3 artefacts — Q_T bound for a given gate set + sample + t.
# ---------------------------------------------------------------------------

class CodeRecord(BaseModel):
    """A quantum error-correcting code and its transversal structure.

    `transversal_groups` lists SWIFTbot registry names (e.g. 'clifford',
    'BI') whose action is realised transversally on this code — meaning a
    run of SWIFTbot that picked one of those groups as its base C has a
    candidate fault-tolerant realisation here.

    Provenance should cite the paper that established the transversality.
    Records marked `research_needed=True` are speculative — no concrete
    paper confirmation yet, e.g. entries synthesised from Error Correction
    Zoo browsing."""

    name: str                                # e.g. "Steane [[7,1,3]]"
    n: int | None = Field(default=None, ge=0)         # physical qudits (None = family; parameters vary)
    k: int | None = Field(default=None, ge=0)
    distance: int | None = Field(default=None, ge=0)
    qudit_dim: int = Field(..., gt=0)        # 2 for qubit, 3 for qutrit, ...
    transversal_groups: list[str] = Field(default_factory=list)
    stabilizer: bool = True
    reference: str = ""                      # arXiv / DOI / URL
    notes: str = ""
    research_needed: bool = False            # speculative entry?


class DistillationRecord(BaseModel):
    """A magic-state distillation protocol.

    `target_gate_family` is a coarse string tag — 'qubit T', 'qubit CCZ',
    'qutrit T' (Howard-Vala), 'qutrit strange'. Matching against an
    ExtensionSpec is by tag, not exact matrix identity.

    `yield_parameter` is the protocol's γ (exponent in log-log overhead
    scaling): lower is better. None when not quantitatively reported.
    """

    protocol_name: str                       # e.g. "Bravyi-Kitaev 15-to-1"
    target_gate_family: str                  # coarse tag
    qudit_dim: int = Field(..., gt=0)
    code_name: str                           # which code hosts it
    yield_parameter: float | None = None     # γ in O(log^γ(1/ε)) overhead
    reference: str = ""
    notes: str = ""
    research_needed: bool = False


class PerTargetCoverage(BaseModel):
    """Distance and synthesis-cost summary for a single target.

    `depth_first_hit`: total word length (≤ max_depth) at which this target
        was first brought within ε_hit.
    `t_count_first_hit`: number of T / T† generator uses in that word — a
        direct fault-tolerance cost proxy (T gates are the expensive ones
        for stabilizer-code implementations). For base groups other than
        Clifford or extensions other than P(π/4), T-count interpretation
        depends on what counts as the "costly" generator; see the
        `method` / `cost_method` fields on the parent record for semantics.
    Both fields are None when the target never came within ε_hit at the
    BFS cap.
    """

    label: str
    distance: float = Field(..., ge=0.0)
    depth_first_hit: int | None = None
    t_count_first_hit: int | None = None


SynthesisCostMethod = Literal[
    "bfs_estimate",            # our BFS: nearest-word depth + T-count in that word
    "ross_selinger_exact",     # arXiv:0912.0917, exact optimal for Clifford + P(π/4)
    "kliuchnikov_maslov_mosca", # 1206.5236, exact Clifford+T normal form
    "sk_asymptotic_bound",     # generic Solovay-Kitaev upper bound
]


class CoverageRecord(BaseModel):
    """Outcome of a target-family coverage evaluation — the Scope-B analogue
    of QTRecord, but for the specialised/non-universal regime.

    `base_group_key`: group_key of the Clifford-like base (from `groups` table).
    `ext_fingerprint`: sha256 digest of the ExtensionSpec (see s3_efficiency).
    `target_family_name`: name of a registered `TargetFamily` (e.g. 'lamm_sigma36').
    `max_depth`: BFS cap used.
    `per_target`: per-target distance/depth; length must equal `n_targets`.

    `cost_method`: HOW the T-counts were obtained. 'bfs_estimate' is the
        default and means per-target t_count comes from tracking T/T† uses
        along the specific BFS path that reached ε; this is *an* upper bound,
        not a proven minimum. For Clifford + P(π/4) specifically there exist
        certified exact normal forms (Ross-Selinger, Kliuchnikov-Maslov-Mosca)
        — swap those in to get `ross_selinger_exact` etc. with `certified=True`.
        For arbitrary extensions (e.g. P(2π/9), super-golden, Haar-random) no
        normal-form theorem is proved in general; treat t_count as an estimate.
    `certified`: True iff the reported T-counts are proved optimal. Always
        False for 'bfs_estimate'. Intended for downstream resource accounting.
    `cost_method_notes`: free-text provenance (paper citation, caveat, etc.).
    """

    base_group_key: str
    ext_fingerprint: str = ""
    target_family_name: str
    max_depth: int = Field(..., gt=0)
    n_targets: int = Field(..., gt=0)
    visited: int = Field(..., gt=0)
    mean_dist: float = Field(..., ge=0.0)
    max_dist: float = Field(..., ge=0.0)
    hits_count: int = Field(..., ge=0)
    mean_t_count_hits: float | None = None   # mean T-count over hit targets; None if no hits
    per_target: list[PerTargetCoverage] = Field(default_factory=list)
    cost_method: SynthesisCostMethod = "bfs_estimate"
    certified: bool = False
    cost_method_notes: str = ""


class QTRecord(BaseModel):
    """One (t, sample_id) measurement of δ and Q_T on a target set.

    `target_key` is the *base* group's content hash (from the `groups` table).
    `ext_fingerprint` distinguishes different extensions applied to that
    base — so that `(clifford + rnd, t=5, sample_id=0)` and
    `(clifford + angle(π/4), t=5, sample_id=0)` are stored as separate rows.
    """

    target_key: str                          # BASE group's group_key
    t: int = Field(..., gt=0)
    sample_id: int = 0                       # for random-extension ensembles
    delta: float                             # δ(ν_S, t), max norm over weights
    qt: float | None = None                  # Q_T = log|C| / log(1/δ); None if ill-defined
    q_opt: float | None = None               # Kesten-McKay optimum for |C|
    source_file: str | None = None           # original qcoG*.txt path if applicable
    ext_fingerprint: str = ""                # stable hash of the extension spec; "" for the base alone
