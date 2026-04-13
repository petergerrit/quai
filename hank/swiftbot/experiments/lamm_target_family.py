"""Scope-A test case: coverage of the Lamm Σ(36×3) target family.

Quick numerical kernel (no agent, no SWIFTbot sweep) for the question:
given a d=2 Clifford base and a candidate extension T, how well does the
word tree ⟨Clifford, T, T†⟩ cover the target family

    T_discrete  = { R_Z(2π k/9) : k = 1, 2, …, 8 }
    T_continuous = { R_Z(k · θ) : k ∈ {1, 3, 5, 9, 15}, θ ∈ (0, 2π) }

which combines U_F's discrete 9th-root phases and U_Tr's continuous
parametric family from the Lamm/Gustafson Σ(36×3) primitive-gate paper.

Method: bounded-depth BFS over generating set {H, S, S†, T, T†}, dedup by
projective-matrix hash (ignore global U(1) phase). For each target, record
the minimum ε-distance achieved and the word depth at which it was first
reached within a given tolerance.

Run:

    swiftbot/.venv/bin/python -m swiftbot.experiments.lamm_target_family

Tuning:
    MAX_DEPTH   — BFS cap. 10 explores ~O(10^4) unique projective words.
    N_THETA     — number of continuous θ samples to average over.
    EPS_HIT     — tolerance for counting a target as "reached".
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from swiftbot.tools import groups as gmod  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_DEPTH = 10
N_THETA_SAMPLES = 10
CONTINUOUS_K = (1, 3, 5, 9, 15)
EPS_HIT = 1e-2        # "close enough" for depth_at_first_hit
MAX_UNIQUE = 200_000  # safety cap on the word tree size
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Basic matrices + keys
# ---------------------------------------------------------------------------

def rz(phi: float) -> np.ndarray:
    """SU(2) R_Z(φ) = diag(e^{-iφ/2}, e^{iφ/2})."""
    return np.array(
        [[np.exp(-1j * phi / 2), 0.0],
         [0.0, np.exp(1j * phi / 2)]],
        dtype=complex,
    )


H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdag = S_gate.conj().T


def _normalize_sud(M: np.ndarray) -> np.ndarray:
    det = np.linalg.det(M)
    d = M.shape[0]
    return M / det ** (1.0 / d)


# Hurwitz group generators (SU(2)-normalised), per qco-main/finite_groups.py.
# Closure is the 12-element A4 ⊂ PU(2) lift (|C| = 12 projectively; 24 as SU(2)).
_H_raw1 = np.array([[1j, 0], [0, -1j]], dtype=complex)          # already SU(2)
_H_raw2 = np.array([[1, 1], [1j, -1j]], dtype=complex)
HURWITZ_GENS = [_normalize_sud(_H_raw1), _normalize_sud(_H_raw2),
                _normalize_sud(_H_raw2).conj().T]

CLIFFORD_GENS = [H_gate, S_gate, Sdag]


def super_golden_T_24() -> np.ndarray:
    """Sarnak-Parzanchevski super-golden gate T_24 for Clifford completion
    (arXiv:1704.02106). Defined 'up to normalisation'; we SU(2)-normalise."""
    raw = np.array([
        [-1 - np.sqrt(2), 2 - np.sqrt(2) + 1j],
        [2 - np.sqrt(2) - 1j, 1 + np.sqrt(2)],
    ], dtype=complex)
    return _normalize_sud(raw)


def super_golden_T_12() -> np.ndarray:
    """Super-golden gate T_12 for Hurwitz completion (arXiv:1704.02106)."""
    raw = np.array([[3, 1 - 1j], [1 + 1j, -3]], dtype=complex)
    return _normalize_sud(raw)


def _projective_key(M: np.ndarray, decimals: int = 6) -> bytes:
    """Hash ignoring global U(1) phase: rotate first non-zero entry to be
    real positive, then hash rounded real+imag arrays."""
    flat = M.ravel()
    # Find first element with nontrivial magnitude (cheap index scan)
    for i in range(flat.size):
        z = flat[i]
        if abs(z) > 1e-9:
            phase = z / abs(z)
            C = M * phase.conj()
            break
    else:
        C = M
    re = np.round(C.real, decimals)
    im = np.round(C.imag, decimals)
    # Kill -0.0 / +0.0 ambiguity
    re += 0.0
    im += 0.0
    return hashlib.sha1(re.tobytes() + im.tobytes()).digest()


def projective_distance(A: np.ndarray, B: np.ndarray) -> float:
    """min over global phase φ of ||A - e^{iφ} B||_op (for SU(2) unitaries).

    Closed form (SU(2), both unitary): Fubini-Study / Frobenius-derived,
        d² = 2 · (1 − |tr(A† B)| / 2)
    So d = sqrt(2 − |tr(A† B)|). Much faster than a 256-point phase sweep.
    """
    inner = np.vdot(A.ravel(), B.ravel())    # tr(A† B)
    mag = abs(inner)
    val = 2.0 - mag
    return math.sqrt(max(val, 0.0))


def projective_distances_batched(M: np.ndarray, T_stack: np.ndarray) -> np.ndarray:
    """Vectorised projective_distance of a single M against a stack of targets.

    T_stack has shape (n_targets, 2, 2). Returns shape (n_targets,).
    """
    Mdag = M.conj().T
    # inner[i] = trace(M† · T_stack[i]) = sum_jk (M†)[j,k] T[i,k,j]
    inner = np.einsum("jk,ikj->i", Mdag, T_stack)
    mag = np.abs(inner)
    return np.sqrt(np.maximum(2.0 - mag, 0.0))


# ---------------------------------------------------------------------------
# Target family
# ---------------------------------------------------------------------------

@dataclass
class Target:
    label: str
    matrix: np.ndarray


def lamm_target_family(
    n_theta: int = N_THETA_SAMPLES,
    rng_seed: int = RNG_SEED,
    variant: str = "sigma36",
) -> list[Target]:
    """Lamm-style target family for an SU(3) discrete-subgroup lattice simulation.

    variant = 'sigma36' : 8 discrete 9th-root phases + 50 continuous R_Z(k·θ).
    variant = 'sigma72' : the above PLUS 8 discrete 18th-root phases
                          (Σ(72×3) irreps include ω_18 structure beyond ω_9).
    """
    rng = np.random.default_rng(rng_seed)
    ts: list[Target] = []
    for k in range(1, 9):
        ts.append(Target(label=f"discrete R_Z(2π·{k}/9)", matrix=rz(2 * np.pi * k / 9)))
    if variant == "sigma72":
        # Σ(72×3) picks up 6th-root structure (±1/2 ∈ Re Tr from the V₂-extension)
        # which together with 9th-roots gives ω_18.
        for k in range(1, 18):
            if 18 % k != 0 or k == 1:  # skip duplicates of 9th-roots/identity
                ts.append(Target(
                    label=f"discrete R_Z(2π·{k}/18)",
                    matrix=rz(2 * np.pi * k / 18),
                ))
    thetas = rng.uniform(0.05, np.pi - 0.05, size=n_theta)
    for i, theta in enumerate(thetas):
        for k in CONTINUOUS_K:
            ts.append(Target(
                label=f"cont R_Z({k}·θ) θ_idx={i} θ={theta:.3f}",
                matrix=rz(k * theta),
            ))
    return ts


# ---------------------------------------------------------------------------
# Word-tree BFS
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult:
    extension_name: str
    min_dists: list[float]
    depths_reached: list[int | None]
    visited: int


def bfs_coverage(
    generators: list[np.ndarray],
    targets: list[Target],
    max_depth: int = MAX_DEPTH,
    eps_hit: float = EPS_HIT,
    max_unique: int = MAX_UNIQUE,
    verbose: bool = False,
) -> CoverageResult:
    d = generators[0].shape[0]
    I = np.eye(d, dtype=complex)
    T_stack = np.asarray([t.matrix for t in targets], dtype=complex)   # (n, 2, 2)
    n_targets = T_stack.shape[0]

    min_dists = np.full(n_targets, np.inf, dtype=float)
    depths_reached: list[int | None] = [None] * n_targets

    # Prime with identity
    seen = {_projective_key(I)}
    frontier = [I]
    total_visited = 1

    d_I = projective_distances_batched(I, T_stack)
    better = d_I < min_dists
    min_dists = np.where(better, d_I, min_dists)

    for depth in range(1, max_depth + 1):
        new_frontier: list[np.ndarray] = []
        new_seen: set[bytes] = set()
        for M in frontier:
            for g in generators:
                new_M = g @ M
                k = _projective_key(new_M)
                if k in seen or k in new_seen:
                    continue
                new_seen.add(k)
                new_frontier.append(new_M)
                # Score against targets (vectorised)
                dists = projective_distances_batched(new_M, T_stack)
                improved = dists < min_dists
                if improved.any():
                    min_dists = np.where(improved, dists, min_dists)
                    for i in np.nonzero(improved)[0]:
                        if (
                            depths_reached[int(i)] is None
                            and min_dists[int(i)] <= eps_hit
                        ):
                            depths_reached[int(i)] = depth
                if total_visited + len(new_seen) >= max_unique:
                    break
            if total_visited + len(new_seen) >= max_unique:
                break
        seen.update(new_seen)
        frontier = new_frontier
        total_visited += len(new_seen)
        if verbose:
            hits = sum(1 for x in depths_reached if x is not None)
            print(f"    depth={depth}  visited={total_visited}  hits={hits}/{n_targets}  "
                  f"mean_min={float(np.mean(min_dists)):.4f}", flush=True)
        if not new_frontier or total_visited >= max_unique:
            break

    return CoverageResult(
        extension_name="",
        min_dists=list(min_dists),
        depths_reached=depths_reached,
        visited=total_visited,
    )


# ---------------------------------------------------------------------------
# Candidate extensions
# ---------------------------------------------------------------------------

def haar_su2(rng: np.random.Generator) -> np.ndarray:
    # Haar-random SU(2) via ZYZ angles (Ozols 2009 method).
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    ct = rng.uniform(-1, 1)
    st = math.sqrt(1 - ct ** 2)
    U = np.array(
        [[math.cos(a) * ct, -math.sin(a) * st],
         [math.sin(a) * st, math.cos(a) * ct]],
        dtype=complex,
    )
    # Phase perturbation to randomise off-diagonal too
    U2 = np.diag([np.exp(1j * b), np.exp(-1j * b)]) @ U @ np.diag([np.exp(1j * c), np.exp(-1j * c)])
    det = np.linalg.det(U2)
    return U2 / det ** 0.5


def candidate_extensions(
    rng_seed: int = RNG_SEED,
    include_super_golden: bool = True,
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(rng_seed + 1)
    cands: list[tuple[str, np.ndarray]] = [
        ("T = P(π/4) [canonical]", rz(np.pi / 4)),
        ("T = P(π/8)",              rz(np.pi / 8)),
        ("T = P(π/9)",              rz(np.pi / 9)),
        ("T = P(2π/9)",             rz(2 * np.pi / 9)),
        ("T = P(π/18)",             rz(np.pi / 18)),
        ("T = Haar-random #1",      haar_su2(rng)),
        ("T = Haar-random #2",      haar_su2(rng)),
    ]
    if include_super_golden:
        cands.append(("T_24 = super-golden (Clifford)", super_golden_T_24()))
        cands.append(("T_12 = super-golden (Hurwitz)",  super_golden_T_12()))
    return cands


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(
    max_depth: int = MAX_DEPTH,
    n_theta: int = N_THETA_SAMPLES,
    base: str = "clifford",
    variant: str = "sigma36",
) -> None:
    targets = lamm_target_family(n_theta=n_theta, variant=variant)
    candidates = candidate_extensions()

    print(f"Lamm-{variant} target family — Scope-A coverage test", flush=True)
    print(f"  base: {base}  targets: {len(targets)}  BFS depth≤{max_depth}  "
          f"max_unique={MAX_UNIQUE}  eps={EPS_HIT}")
    print()

    if base == "clifford":
        base_gens = CLIFFORD_GENS
    elif base == "hurwitz":
        base_gens = HURWITZ_GENS
    else:
        raise SystemExit(f"unknown base: {base}")
    clifford_gens = base_gens  # local alias retained

    results: list[CoverageResult] = []
    for name, T in candidates:
        print(f"  exploring extension: {name}", flush=True)
        generators = clifford_gens + [T, T.conj().T]
        t0 = time.time()
        r = bfs_coverage(generators, targets, max_depth=max_depth, verbose=True)
        dt = time.time() - t0
        r.extension_name = name
        results.append(r)

        hits = sum(1 for d in r.depths_reached if d is not None)
        print(f"    visited={r.visited:>6d}  targets_hit_under_eps={hits}/{len(targets)}  "
              f"mean_min_dist={float(np.mean(r.min_dists)):.4f}  time={dt:.1f}s",
              flush=True)

    # --- Summary table
    print()
    print("=" * 92)
    print(f"{'extension':<28}  {'visited':>8}  {'mean_min_d':>10}  "
          f"{'max_min_d':>10}  {'hits_ε':>8}  {'mean_depth_hit':>14}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: float(np.mean(x.min_dists))):
        hits = [d for d in r.depths_reached if d is not None]
        mean_depth = float(np.mean(hits)) if hits else float("nan")
        print(f"{r.extension_name:<28}  {r.visited:>8d}  {float(np.mean(r.min_dists)):>10.4f}  "
              f"{float(np.max(r.min_dists)):>10.4f}  {len(hits):>4}/{len(targets):<3}  "
              f"{mean_depth:>14.2f}")
    print("=" * 92)

    # --- Split report: discrete vs continuous
    disc_idx = [i for i, t in enumerate(targets) if t.label.startswith("discrete")]
    cont_idx = [i for i, t in enumerate(targets) if t.label.startswith("cont")]
    print()
    print("Split:  discrete 9th-root targets vs continuous R_Z(kθ) targets")
    print(f"{'extension':<28}  {'disc mean d':>12}  {'disc hits':>10}  "
          f"{'cont mean d':>12}  {'cont hits':>10}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: float(np.mean([x.min_dists[i] for i in disc_idx]))):
        dm = float(np.mean([r.min_dists[i] for i in disc_idx]))
        dh = sum(1 for i in disc_idx if r.depths_reached[i] is not None)
        cm = float(np.mean([r.min_dists[i] for i in cont_idx]))
        ch = sum(1 for i in cont_idx if r.depths_reached[i] is not None)
        print(f"{r.extension_name:<28}  {dm:>12.4f}  {dh:>4}/{len(disc_idx):<3}  "
              f"{cm:>12.4f}  {ch:>4}/{len(cont_idx):<3}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    p.add_argument("--n-theta", type=int, default=N_THETA_SAMPLES)
    p.add_argument("--base", default="clifford", choices=("clifford", "hurwitz"))
    p.add_argument("--variant", default="sigma36", choices=("sigma36", "sigma72"))
    args = p.parse_args()
    run(max_depth=args.max_depth, n_theta=args.n_theta,
        base=args.base, variant=args.variant)
