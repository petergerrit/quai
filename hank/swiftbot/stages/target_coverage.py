"""Stage-B coverage evaluator: word-tree BFS against a registered TargetFamily.

Sibling of Stage 3 (Q_T) — both score a (base_group, extension) pair, but
against different things:
    * Stage 3 (qco.py / s3_efficiency): Q_T — generic SU(d) coverage.
    * This module:                       CoverageRecord — a specific,
                                         user-registered `TargetFamily`.

Algorithm: bounded-depth BFS on generators = base_generating_set ∪ {T, T†};
projective-matrix hash for dedup; closed-form SU(2) phase-optimised distance
`d = sqrt(2 − |tr(A† B)|)`. Ported from the Scope-A script
`swiftbot/experiments/lamm_target_family.py`.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from swiftbot.kb.cache import Cache
from swiftbot.state import CoverageRecord, PerTargetCoverage, SynthesisCostMethod
from swiftbot.targets import TargetFamily, get_target_family
from swiftbot.tools import groups as gmod


# ---------------------------------------------------------------------------
# Cost-method catalog — maps (base, ext_kind, ext_params_key) → certified method
# ---------------------------------------------------------------------------
# When a (base, extension) pair has a PROVEN optimal T-count algorithm in the
# literature, we record it here so the caller can (manually) swap the
# bfs_estimate for the certified method. Listing presence does NOT mean the
# pipeline invokes that method automatically — it only signals a citation.

_KNOWN_EXACT_METHODS: dict[tuple[str, str], tuple[SynthesisCostMethod, str]] = {
    # Canonical qubit Clifford+T: Ross-Selinger 2016, Kliuchnikov-Maslov-Mosca 2013.
    ("clifford", "P(π/4)"): (
        "ross_selinger_exact",
        "Ross & Selinger 2016 (arXiv:0912.0917) give an exact optimal "
        "Clifford+T synthesis for z-rotations with T-count 3·log₂(1/ε) + O(log log). "
        "Kliuchnikov-Maslov-Mosca 2013 (arXiv:1206.5236) give the ring-based "
        "Clifford+T normal form. bfs_estimate here is an upper bound; swap in "
        "those algorithms for certified-optimal T-counts.",
    ),
}


def certified_method_for(base_name: str, extension_label: str) -> tuple[SynthesisCostMethod, str] | None:
    """If the (base, extension) pair has a published exact normal-form
    algorithm, return (method_tag, citation). Otherwise None.

    Non-presence means "no published normal form we know of" — the
    bfs_estimate remains the best available method, treated as an upper
    bound on the optimal T-count.
    """
    return _KNOWN_EXACT_METHODS.get((base_name, extension_label))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "clifford_t"))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_DEPTH = 10
DEFAULT_N_PARAMETRIC = 10
DEFAULT_EPS_HIT = 1e-2
DEFAULT_MAX_UNIQUE = 200_000


# ---------------------------------------------------------------------------
# Hash + distance primitives
# ---------------------------------------------------------------------------

def _projective_key(M: np.ndarray, decimals: int = 6) -> bytes:
    flat = M.ravel()
    for i in range(flat.size):
        z = flat[i]
        if abs(z) > 1e-9:
            phase = z / abs(z)
            C = M * phase.conj()
            break
    else:
        C = M
    re = np.round(C.real, decimals) + 0.0
    im = np.round(C.imag, decimals) + 0.0
    return hashlib.sha1(re.tobytes() + im.tobytes()).digest()


def _projective_distance_batched(M: np.ndarray, T_stack: np.ndarray) -> np.ndarray:
    """min over global phase φ of ||M − e^{iφ} T_stack[i]||_op for each i.

    Closed form for SU(2) unitaries: d = sqrt(2 − |tr(M† T)|). The einsum
    below computes tr(M† T[i]) = ⟨M, T[i]⟩_F for each i at once.
    """
    Mdag = M.conj().T
    inner = np.einsum("jk,ikj->i", Mdag, T_stack)
    mag = np.abs(inner)
    return np.sqrt(np.maximum(2.0 - mag, 0.0))


# ---------------------------------------------------------------------------
# BFS coverage kernel
# ---------------------------------------------------------------------------

def evaluate_coverage(
    base_generators: Sequence[np.ndarray],
    extension: np.ndarray,
    targets: list[tuple[str, np.ndarray]],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    eps_hit: float = DEFAULT_EPS_HIT,
    max_unique: int = DEFAULT_MAX_UNIQUE,
) -> dict:
    """Pure kernel: BFS word tree vs a concrete target list.

    Each BFS node is a (matrix, depth, t_count) triple where t_count is the
    number of times the last two generators (T, T†) were used along the
    path that reached this matrix. When a target is first ε-approximated
    we record that word's (depth, t_count). This is an UPPER BOUND on the
    optimal T-count; for certified optimality on Clifford + P(π/4) swap in
    Ross-Selinger (arXiv:0912.0917) or Kliuchnikov-Maslov-Mosca
    (arXiv:1206.5236).

    Returns dict: n_targets, visited, mean_dist, max_dist, hits_count,
    labels, per_target_distances, per_target_depth_first_hit,
    per_target_t_count_first_hit, mean_t_count_hits.
    """
    if not targets:
        raise ValueError("targets list is empty")
    d = base_generators[0].shape[0]
    # Generators: base first, then T, T†. The last two are the "costly" ones.
    base_list = list(base_generators)
    generators = base_list + [extension, extension.conj().T]
    n_base = len(base_list)
    # Bool mask: True for T/T† generators (the "costly" ones we count).
    gen_is_costly = np.array(
        [False] * n_base + [True, True], dtype=bool,
    )

    I = np.eye(d, dtype=complex)
    labels = [t[0] for t in targets]
    T_stack = np.asarray([t[1] for t in targets], dtype=complex)

    min_dists = np.full(T_stack.shape[0], np.inf, dtype=float)
    depths: list[int | None] = [None] * T_stack.shape[0]
    t_counts: list[int | None] = [None] * T_stack.shape[0]

    seen = {_projective_key(I)}
    # Frontier items: (matrix, t_count_along_path)
    frontier: list[tuple[np.ndarray, int]] = [(I, 0)]
    visited = 1

    d_I = _projective_distance_batched(I, T_stack)
    min_dists = np.minimum(min_dists, d_I)

    for depth in range(1, max_depth + 1):
        new_frontier: list[tuple[np.ndarray, int]] = []
        new_seen: set[bytes] = set()
        for M, tc in frontier:
            for gi, g in enumerate(generators):
                new_M = g @ M
                k = _projective_key(new_M)
                if k in seen or k in new_seen:
                    continue
                new_tc = tc + (1 if gen_is_costly[gi] else 0)
                new_seen.add(k)
                new_frontier.append((new_M, new_tc))
                dists = _projective_distance_batched(new_M, T_stack)
                improved = dists < min_dists
                if improved.any():
                    min_dists = np.where(improved, dists, min_dists)
                    for i in np.nonzero(improved)[0]:
                        i = int(i)
                        if depths[i] is None and min_dists[i] <= eps_hit:
                            depths[i] = depth
                            t_counts[i] = new_tc
                if visited + len(new_seen) >= max_unique:
                    break
            if visited + len(new_seen) >= max_unique:
                break
        seen.update(new_seen)
        frontier = new_frontier
        visited += len(new_seen)
        if not new_frontier or visited >= max_unique:
            break

    hit_tcs = [tc for tc in t_counts if tc is not None]
    mean_tc = float(np.mean(hit_tcs)) if hit_tcs else None

    return {
        "n_targets": len(labels),
        "visited": visited,
        "mean_dist": float(np.mean(min_dists)),
        "max_dist":  float(np.max(min_dists)),
        "hits_count": sum(1 for x in depths if x is not None),
        "mean_t_count_hits": mean_tc,
        "labels": labels,
        "per_target_distances": [float(x) for x in min_dists],
        "per_target_depth_first_hit": depths,
        "per_target_t_count_first_hit": t_counts,
    }


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def evaluate_coverage_by_name(
    base_group_name: str,
    extension_matrix: np.ndarray,
    target_family_name: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    n_parametric_samples: int = DEFAULT_N_PARAMETRIC,
    eps_hit: float = DEFAULT_EPS_HIT,
    max_unique: int = DEFAULT_MAX_UNIQUE,
    rng_seed: int = 0,
    cache: Cache | None = None,
    ext_fingerprint: str = "",
) -> CoverageRecord:
    """Full resolver: look up base generators + target family, run BFS,
    optionally persist to cache. Returns a CoverageRecord."""
    base_generators = gmod.get_generating_set(base_group_name)
    family = get_target_family(target_family_name)
    targets = family.materialize(
        n_parametric_samples=n_parametric_samples, rng_seed=rng_seed,
    )
    kernel = evaluate_coverage(
        base_generators, extension_matrix, targets,
        max_depth=max_depth, eps_hit=eps_hit, max_unique=max_unique,
    )

    # Resolve base_group_key from the registry matrices (content-hash).
    from swiftbot.kb.cache import matrices_key
    base_full_mats = np.asarray(gmod.get_group(base_group_name))
    base_key = matrices_key(base_full_mats)

    per_target = [
        PerTargetCoverage(
            label=kernel["labels"][i],
            distance=kernel["per_target_distances"][i],
            depth_first_hit=kernel["per_target_depth_first_hit"][i],
            t_count_first_hit=kernel["per_target_t_count_first_hit"][i],
        )
        for i in range(kernel["n_targets"])
    ]
    record = CoverageRecord(
        base_group_key=base_key,
        ext_fingerprint=ext_fingerprint,
        target_family_name=target_family_name,
        max_depth=max_depth,
        n_targets=kernel["n_targets"],
        visited=kernel["visited"],
        mean_dist=kernel["mean_dist"],
        max_dist=kernel["max_dist"],
        hits_count=kernel["hits_count"],
        mean_t_count_hits=kernel["mean_t_count_hits"],
        per_target=per_target,
        cost_method="bfs_estimate",
        certified=False,
        cost_method_notes=(
            "BFS upper bound; see certified_method_for(base, extension) for "
            "pairs with published exact normal forms (Ross-Selinger et al.)."
        ),
    )

    if cache is not None:
        # Ensure the base group is registered so the FK-like join holds.
        cache.put_group(
            base_full_mats,
            name=base_group_name,
            source=f"registry:{base_group_name}",
            projective=gmod.REGISTRY[base_group_name].projective,
        )
        cache.put_coverage(record)

    return record
