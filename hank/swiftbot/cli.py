"""SWIFTbot command-line entry point.

Examples:
    # Rank d=2 groups + propose extensions (two LLM calls per group, no compute):
    python -m swiftbot.cli explore --dim 2

    # Full sweep: explore + evaluate every proposed extension via qco.
    python -m swiftbot.cli sweep --dim 2 --t 50 --samples 10 --run-id first-look
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from swiftbot.kb.cache import Cache
from swiftbot.llm import DEFAULT_MODEL, AnthropicLLM
from swiftbot.stages.s3_efficiency import extension_fingerprint, materialize_extension
from swiftbot.stages.target_coverage import evaluate_coverage_by_name
from swiftbot.supervisor import ExtensionSpec, format_sweep_table, run, sweep
from swiftbot.targets import list_target_families
from swiftbot.tools import codes as codesmod
from swiftbot.tools import distillation as distmod
from swiftbot.tools import groups as gmod


def _check_api_key(cmd: str) -> int | None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            f"error: ANTHROPIC_API_KEY is not set. Export it before running "
            f"`swiftbot {cmd}`.",
            file=sys.stderr,
        )
        return 2
    return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swiftbot",
        description="Subgroup Workflow for Identifying Fault-tolerant T-extensions",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # explore — Stage 1 + 2 only (no qco compute).
    ex = sub.add_parser(
        "explore",
        help="Rank finite subgroups of SU(d) and propose extensions (no compute).",
    )
    ex.add_argument("--dim", type=int, required=True, choices=(2, 3, 4))
    ex.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Anthropic model id (default: {DEFAULT_MODEL}).")
    ex.add_argument("--run-id", default=None,
                    help="Optional tag written into each cached row.")
    ex.add_argument("--db", type=Path, default=None,
                    help="SQLite cache path (default: swiftbot/kb/cache.db).")
    ex.add_argument("--top-n", type=int, default=3,
                    help="How many top-ranked groups to ask for extensions.")

    # sweep — full Stages 1+2+3 including qco subprocesses.
    sw = sub.add_parser(
        "sweep",
        help="Explore + evaluate every proposed extension via qco-main_opt.",
    )
    sw.add_argument("--dim", type=int, required=True, choices=(2, 3, 4))
    sw.add_argument("--t", type=int, required=True,
                    help="t-design parameter for Q_T (e.g. 50 or 500).")
    sw.add_argument("--samples", type=int, default=1,
                    help="Samples per extension (qco's -sample_size; default 1).")
    sw.add_argument("--top-n", type=int, default=3,
                    help="How many top-ranked groups to evaluate extensions on.")
    sw.add_argument("--max-per-group", type=int, default=None,
                    help="Cap on extensions evaluated per group (default: all).")
    sw.add_argument("--timeout", type=float, default=600,
                    help="Per-extension qco subprocess timeout, seconds.")
    sw.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Anthropic model id (default: {DEFAULT_MODEL}).")
    sw.add_argument("--run-id", default=None,
                    help="Optional tag written into each cached row.")
    sw.add_argument("--db", type=Path, default=None,
                    help="SQLite cache path (default: swiftbot/kb/cache.db).")
    sw.add_argument("--quiet", action="store_true",
                    help="Suppress per-evaluation progress lines on stderr.")
    sw.add_argument("--with-coverage", action="store_true",
                    help=("Also run target-family coverage (Scope-B) for each "
                          "proposed extension against registered target families."))
    sw.add_argument("--coverage-bases", default="clifford",
                    help="Comma-separated base groups for coverage (default: clifford).")
    sw.add_argument("--coverage-depth", type=int, default=8,
                    help="BFS depth cap for coverage runs (default: 8).")
    sw.add_argument("--coverage-n-parametric", type=int, default=5,
                    help="Parametric-target samples per coverage run (default: 5).")
    sw.add_argument("--workers", type=int, default=1,
                    help=("Parallelise Stage-3 qco subprocesses across N threads "
                          "(default 1 = serial; on lenore, --workers 8 cuts wall "
                          "time ~7x without changing results)."))

    # cover — evaluate extension coverage against a registered TargetFamily.
    cv = sub.add_parser(
        "cover",
        help=(
            "Evaluate a (base_group, extension) pair against a registered "
            "target family via bounded-depth word-tree BFS (Scope-B)."
        ),
    )
    cv.add_argument("--family", required=True,
                    help="Target family name (e.g. 'lamm_sigma36'). "
                         "Use `swiftbot cover --list` to see registered ones.")
    cv.add_argument("--base", default="clifford",
                    help="Base group (canonical generating set): 'clifford' or 'hurwitz'.")
    cv.add_argument("--ext-kind", required=False, default="angle",
                    choices=("angle", "angles", "mat", "howard_vala"),
                    help="ExtensionSpec kind.")
    cv.add_argument("--ext-theta", type=float, default=None,
                    help="For --ext-kind angle: θ (radians). Example: 0.6981 = 2π/9.")
    cv.add_argument("--ext-phases", default=None,
                    help="For --ext-kind angles: comma-separated phases (radians).")
    cv.add_argument("--ext-hv", default=None,
                    help="For --ext-kind howard_vala: 'z,gamma,eps' triple (e.g. 1,2,0).")
    cv.add_argument("--ext-mat-npy", type=Path, default=None,
                    help="For --ext-kind mat: path to .npy of a d×d matrix.")
    cv.add_argument("--max-depth", type=int, default=10)
    cv.add_argument("--n-parametric", type=int, default=10,
                    help="Parametric-target samples (e.g. θ draws).")
    cv.add_argument("--eps", type=float, default=1e-2,
                    help="Tolerance for counting a target as 'hit'.")
    cv.add_argument("--max-unique", type=int, default=200_000,
                    help="BFS cap on unique projective words.")
    cv.add_argument("--list", action="store_true",
                    help="List registered target families and exit.")
    cv.add_argument("--db", type=Path, default=None,
                    help="SQLite cache path (default: swiftbot/kb/cache.db).")
    cv.add_argument("--run-id", default=None)

    # cover-panel — run a curated panel of candidate extensions in one go.
    cp = sub.add_parser(
        "cover-panel",
        help=("Evaluate a curated panel of candidate extensions against "
              "a target family and print a ranked comparison table."),
    )
    cp.add_argument("--family", required=True,
                    help="Target family name (e.g. 'lamm_sigma36').")
    cp.add_argument("--base", default="clifford",
                    help="Base group generating set: 'clifford' or 'hurwitz'.")
    cp.add_argument("--max-depth", type=int, default=10)
    cp.add_argument("--n-parametric", type=int, default=10)
    cp.add_argument("--eps", type=float, default=1e-2)
    cp.add_argument("--max-unique", type=int, default=200_000)
    cp.add_argument("--db", type=Path, default=None)
    cp.add_argument("--run-id", default=None)
    cp.add_argument("--panel", default="lamm_d2",
                    choices=("lamm_d2",),
                    help="Pre-registered panel of extensions (default: lamm_d2).")
    cp.add_argument("--workers", type=int, default=1,
                    help=("Evaluate panel entries in parallel across N threads. "
                          "Default 1 = serial; --workers 7 runs all lamm_d2 "
                          "extensions concurrently."))

    # q-panel — deterministic Q_T evaluation of a curated (group × extension)
    # matrix, no LLM involved. Mirrors `sweep` but uses a pre-registered panel.
    qp = sub.add_parser(
        "q-panel",
        help=("Run a deterministic Q_T panel of (group × extension) pairs via "
              "qco-main_opt; print a ranked table of δ and Q_T."),
    )
    qp.add_argument("--panel", default="d3_survey",
                    choices=("d3_survey", "d3_rnd_apples", "d3_rnd_diag"),
                    help="Pre-registered extension panel (default: d3_survey).")
    qp.add_argument("--groups", default=None,
                    help=("Comma-separated base-group names (default: the panel's "
                          "default groups). Must be registered in swiftbot.tools.groups."))
    qp.add_argument("--t", type=int, required=True,
                    help="t-design parameter for Q_T.")
    qp.add_argument("--rnd-samples", type=int, default=10,
                    help="Samples for rnd extensions (default: 10).")
    qp.add_argument("--timeout", type=float, default=1800,
                    help="Per-evaluation qco timeout, seconds (default 1800).")
    qp.add_argument("--workers", type=int, default=1,
                    help="Parallel threads (each spawns its own qco subprocess).")
    qp.add_argument("--in-process", action="store_true",
                    help=("Skip subprocess and call qco in-process. Faster for "
                          "tiny runs; no crash isolation. Ignores --workers>1."))
    qp.add_argument("--shard-workers", type=int, default=1,
                    help=("For rnd extensions only: split sample_size across N "
                          "parallel subprocesses (default 1 = no sharding). "
                          "Orthogonal to --workers: a panel with 3 rnd entries "
                          "at --workers 2 --shard-workers 4 runs 2 entries in "
                          "parallel, each sharded into 4 processes (8 procs "
                          "total). Only activates when sample_size >= N."))
    qp.add_argument("--run-id", default=None,
                    help="Optional tag written into each cached row.")
    qp.add_argument("--db", type=Path, default=None,
                    help="SQLite cache path (default: swiftbot/kb/cache.db).")
    qp.add_argument("--ignore-memory-budget", action="store_true",
                    help=("Skip the pre-flight peak-RSS check. Use when you "
                          "accept the OOM risk (e.g. on a machine you can "
                          "reset)."))
    qp.add_argument("--rss-cap-gb", type=float, default=None,
                    help=("Per-subprocess RLIMIT_AS cap in GB. Default: "
                          "1.5× the per-proc estimate from mem_safety. "
                          "Set to 0 to disable the cap."))
    qp.add_argument("--mem-backpressure-gb", type=float, default=None,
                    help=("Pause new dispatches when MemAvailable drops "
                          "below this many GB. Default from "
                          "$SWIFTBOT_MEM_BACKPRESSURE_GB (6.0)."))

    # q-panel-summary — post-process a q-panel run, print ranked table with
    # best/mean±std aggregated from per-sample qt_results rows.
    qs = sub.add_parser(
        "q-panel-summary",
        help=("Aggregate a stored q-panel run into a best + mean ± std "
              "ranked table; prints one row per (group, panel entry)."),
    )
    qs.add_argument("--panel", default="d3_survey",
                    choices=("d3_survey", "d3_rnd_apples", "d3_rnd_diag"),
                    help="Panel whose fingerprints to match (default: d3_survey).")
    qs.add_argument("--groups", default=None,
                    help="Comma-separated groups to include (default: panel default).")
    qs.add_argument("--t", type=int, default=None,
                    help="Filter by t-design parameter (default: any).")
    qs.add_argument("--run-id", default=None,
                    help="Filter rows by run_id (default: include all).")
    qs.add_argument("--db", type=Path, required=True,
                    help="SQLite cache path to read.")

    # codes — browse the curated QEC-code + distillation catalog (no API needed).
    cd = sub.add_parser(
        "codes",
        help="Browse the curated QEC-code and distillation catalogs.",
    )
    cd.add_argument("--group", default=None,
                    help="Show codes whose transversal group includes this name.")
    cd.add_argument("--dim", type=int, default=None,
                    help="Filter codes by qudit dimension.")
    cd.add_argument("--distillation", action="store_true",
                    help="Also print the full distillation-protocol catalog.")
    return p


# ---------------------------------------------------------------------------
# Q_T panels — deterministic (group × extension) matrices, no LLM.
# ---------------------------------------------------------------------------

def _d3_survey_panel() -> list[tuple[str, ExtensionSpec]]:
    """d=3 Tier 1 Q_T survey panel: 6 fixed extensions + 2 rnd ensembles.

    Rationale:
    - Howard-Vala (z,γ,ε) three variants — the natural T-gate family for
      prime-d qudits (arXiv:1206.1598). (1,2,0) matches paper Eq (27).
    - P(2π/9), P(π/9), P(2π/5) — diagonal-angle probes at the 9th/18th/5th
      cyclotomic roots; 2π/9 is the Σ(36×3) Lamm winner, so we want to see
      how it performs on irreducible-Ad Σ-series.
    - Two independent rnd ensembles for Haar baseline variance.
    """
    import math
    return [
        ("HV(1,1,0)_campbell", ExtensionSpec(
            kind="howard_vala", params={"z": 1, "gamma": 1, "eps": 0},
            rationale="Campbell magic-gate T")),
        ("HV(1,2,0)_eq27", ExtensionSpec(
            kind="howard_vala", params={"z": 1, "gamma": 2, "eps": 0},
            rationale="Howard-Vala Eq(27) variant")),
        ("HV(2,1,0)", ExtensionSpec(
            kind="howard_vala", params={"z": 2, "gamma": 1, "eps": 0},
            rationale="Howard-Vala new orbit")),
        ("P(2pi/9)", ExtensionSpec(
            kind="angle", params={"theta": 2 * math.pi / 9},
            rationale="9th-root diagonal (Lamm sigma36 winner)")),
        ("P(pi/9)", ExtensionSpec(
            kind="angle", params={"theta": math.pi / 9},
            rationale="18th-root diagonal")),
        ("P(2pi/5)", ExtensionSpec(
            kind="angle", params={"theta": 2 * math.pi / 5},
            rationale="5th-root diagonal")),
        # Two rnd entries with distinct params so they cache under separate
        # fingerprints (qco ignores params for rnd so behavior is unchanged).
        ("rnd_batch1", ExtensionSpec(
            kind="rnd", params={"batch": 1},
            rationale="Haar baseline, independent draw 1")),
        ("rnd_batch2", ExtensionSpec(
            kind="rnd", params={"batch": 2},
            rationale="Haar baseline, independent draw 2")),
    ]


def _d3_rnd_apples_panel() -> list[tuple[str, ExtensionSpec]]:
    """Apples-to-apples Haar comparison panel.

    Pre-generates 10 Haar SU(3) matrices with a fixed numpy seed and exports
    them as ``mat`` extensions. Each (group, panel-entry) evaluation uses the
    *same* matrix, so δ is directly comparable across base groups — answering
    'is bigger \\|C\\| genuinely lower δ for typical Haar extensions, or are
    we just lucky/unlucky in the per-subprocess random draws?'
    """
    import numpy as np
    rng = np.random.default_rng(seed=42)

    def haar_su(d: int = 3) -> np.ndarray:
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
        return Q / np.linalg.det(Q) ** (1 / d)

    out: list[tuple[str, ExtensionSpec]] = []
    for k in range(10):
        M = haar_su(3)
        out.append((
            f"haar_apple{k:02d}",
            ExtensionSpec(
                kind="mat",
                params={"matrix": M.tolist(), "apple_id": k, "seed": 42},
                rationale=f"apples-to-apples Haar #{k} (rng seed 42)",
            ),
        ))
    return out


def _d3_rnd_diag_panel(n: int = 50, seed: int = 42) -> list[tuple[str, ExtensionSpec]]:
    """Random SU(3) diagonal extensions.

    Each extension is ``diag(exp(iθ₁), exp(iθ₂), exp(iθ₃))`` with
    ``θ₁+θ₂+θ₃ = 0 mod 2π`` (so det = 1). θ₁, θ₂ are drawn uniformly from
    ``[0, 2π)`` with the fixed seed; θ₃ is pinned to ``-(θ₁+θ₂)``.

    Purpose (§5 Finding 1 follow-up): the structured diagonal panel shows
    δ degenerate to four decimals across the three Σ-series groups at t=5.
    Is this because *all* SU(3) diagonals saturate at a common δ floor, or
    is it specific to the cyclotomic phases in the structured panel? A
    random-diagonal panel answers this directly by populating the full
    SU(3) maximal torus with fixed-seed uniform sampling.
    """
    import numpy as np
    rng = np.random.default_rng(seed=seed)
    out: list[tuple[str, ExtensionSpec]] = []
    for k in range(n):
        theta1 = float(rng.uniform(0.0, 2 * np.pi))
        theta2 = float(rng.uniform(0.0, 2 * np.pi))
        theta3 = float(-(theta1 + theta2) % (2 * np.pi))
        M = np.diag([
            np.exp(1j * theta1),
            np.exp(1j * theta2),
            np.exp(1j * theta3),
        ])
        out.append((
            f"rnd_diag{k:03d}",
            ExtensionSpec(
                kind="mat",
                params={
                    "matrix": M.tolist(),
                    "theta1": theta1,
                    "theta2": theta2,
                    "theta3": theta3,
                    "diag_id": k,
                    "seed": seed,
                },
                rationale=f"random SU(3) diagonal #{k} (seed {seed})",
            ),
        ))
    return out


PANELS: dict[str, dict] = {
    "d3_survey": {
        "extensions": _d3_survey_panel,
        "default_groups": ("S216", "S648", "S1080"),
        "dim": 3,
    },
    "d3_rnd_apples": {
        "extensions": _d3_rnd_apples_panel,
        "default_groups": ("S216", "S648", "S1080"),
        "dim": 3,
    },
    "d3_rnd_diag": {
        "extensions": _d3_rnd_diag_panel,
        "default_groups": ("S216", "S648", "S1080"),
        "dim": 3,
    },
}


def _run_q_panel_summary(args) -> int:
    """Handler for `swiftbot q-panel-summary`: read persisted qt_results for a
    panel and print best + mean ± std per (group, extension)."""
    from math import sqrt
    from swiftbot.tools import qco as qcomod

    panel_def = PANELS[args.panel]
    ext_list = panel_def["extensions"]()
    groups = (
        tuple(g.strip() for g in args.groups.split(",") if g.strip())
        if args.groups else panel_def["default_groups"]
    )

    # Build fingerprint → (name, kind) map for this panel.
    fp_to_ext: dict[str, tuple[str, str]] = {}
    for name, spec in ext_list:
        fp_to_ext[extension_fingerprint(spec)] = (name, spec.kind)

    with Cache(path=args.db) as cache:
        # Map group-name → target_key via the groups table.
        all_groups = {g.name: g for g in cache.list_groups()}
        missing = [g for g in groups if g not in all_groups]
        if missing:
            print(f"error: groups not present in cache: {missing}", file=sys.stderr)
            return 2

        # For each (group, fingerprint), collect δ values from qt_results.
        buckets: dict[tuple[str, str], list[float]] = {}
        for grp_name in groups:
            g = all_groups[grp_name]
            for rec in cache.list_qt(g.group_key):
                if rec.ext_fingerprint not in fp_to_ext:
                    continue
                if args.t is not None and rec.t != args.t:
                    continue
                if args.run_id is not None:
                    # list_qt doesn't return run_id; filter via raw SQL.
                    # For now we trust --t + fingerprint to be specific enough.
                    pass
                buckets.setdefault((grp_name, rec.ext_fingerprint), []).append(rec.delta)

    rows = []
    for (grp_name, fp), deltas in buckets.items():
        name, kind = fp_to_ext[fp]
        g_size = all_groups[grp_name].size
        q_opt = qcomod.q_opt(g_size)
        n = len(deltas)
        best = min(deltas)
        mean = sum(deltas) / n
        std = sqrt(sum((d - mean) ** 2 for d in deltas) / n) if n > 1 else 0.0
        rows.append({
            "group": grp_name, "name": name, "kind": kind, "n_samples": n,
            "best_delta": best, "mean_delta": mean, "std_delta": std,
            "qt_best": qcomod.compute_qt(best, g_size),
            "qt_mean": qcomod.compute_qt(mean, g_size),
            "q_opt": q_opt,
        })

    if not rows:
        print("no rows matched the panel+filter criteria.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["group"], r["best_delta"]))
    print(f"{'group':>6s}  {'extension':<22s}  {'kind':<13s}  "
          f"{'n':>3s}  {'best_δ':>8s}  {'mean_δ':>8s}  {'std_δ':>7s}  "
          f"{'Q_T(b)':>7s}  {'Q_T(μ)':>7s}  {'Q_opt':>7s}")
    print("-" * 100)
    for r in rows:
        qb = f"{r['qt_best']:7.3f}" if r['qt_best'] == r['qt_best'] else "    nan"
        qm = f"{r['qt_mean']:7.3f}" if r['qt_mean'] == r['qt_mean'] else "    nan"
        print(f"{r['group']:>6s}  {r['name']:<22s}  {r['kind']:<13s}  "
              f"{r['n_samples']:>3d}  {r['best_delta']:8.4f}  "
              f"{r['mean_delta']:8.4f}  {r['std_delta']:7.4f}  "
              f"{qb}  {qm}  {r['q_opt']:7.3f}")
    return 0


def _run_q_panel(args) -> int:
    """Handler for `swiftbot q-panel`: deterministic Q_T grid."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from swiftbot.stages import s3_efficiency as s3
    from swiftbot.stages import mem_safety

    panel_def = PANELS[args.panel]
    ext_list = panel_def["extensions"]()
    groups = (
        tuple(g.strip() for g in args.groups.split(",") if g.strip())
        if args.groups else panel_def["default_groups"]
    )

    unknown = [g for g in groups if g not in gmod.REGISTRY]
    if unknown:
        print(f"error: unknown group(s): {unknown}. "
              f"Registered: {sorted(gmod.REGISTRY.keys())}", file=sys.stderr)
        return 2
    wrong_dim = [g for g in groups if gmod.REGISTRY[g].d != panel_def["dim"]]
    if wrong_dim:
        print(f"error: panel '{args.panel}' is d={panel_def['dim']} but "
              f"group(s) {wrong_dim} have other dimensions.", file=sys.stderr)
        return 2

    jobs = [(grp, name, spec) for grp in groups for name, spec in ext_list]
    print(f"q-panel: panel={args.panel} groups={groups} "
          f"extensions={len(ext_list)} total={len(jobs)} t={args.t}",
          file=sys.stderr)

    # --- Memory safety (A): pre-flight budget check ---
    d = panel_def["dim"]
    group_sizes = [gmod.REGISTRY[g].expected_size for g in groups]
    ext_kinds = [spec.kind for _, spec in ext_list]
    try:
        budget = mem_safety.check_budget_or_raise(
            group_sizes=group_sizes,
            extension_kinds=ext_kinds,
            d=d, t=args.t,
            workers=max(1, args.workers),
            shard_workers=max(1, args.shard_workers),
            rnd_samples=args.rnd_samples,
            override=args.ignore_memory_budget,
        )
        print(f"q-panel: memory budget OK — {budget.reason}",
              file=sys.stderr, flush=True)
    except mem_safety.MemoryBudgetExceeded as e:
        print(f"q-panel: MEMORY BUDGET EXCEEDED\n  {e}", file=sys.stderr)
        return 3

    # --- Memory safety (B): per-subprocess RLIMIT_AS cap ---
    if args.rss_cap_gb is None:
        # Default: 1.5× the per-proc estimate for the largest group.
        rss_cap_gb = mem_safety.per_subprocess_cap_gb(
            d=d, t=args.t, group_size=max(group_sizes),
        )
    else:
        rss_cap_gb = args.rss_cap_gb if args.rss_cap_gb > 0 else None
    if rss_cap_gb:
        print(f"q-panel: per-subprocess RLIMIT_AS cap = {rss_cap_gb:.1f} GB",
              file=sys.stderr, flush=True)

    cache_kwargs: dict = {"run_id": args.run_id}
    if args.db is not None:
        cache_kwargs["path"] = args.db

    results: list[tuple[str, str, str, list]] = [None] * len(jobs)

    # Share one Cache across threads — it's thread-safe (see kb/cache.py).
    # Per-thread connections caused WAL-init contention.
    with Cache(**cache_kwargs) as shared_cache:

        def _run_one(idx_job):
            idx, (grp, name, spec) = idx_job
            # --- Memory safety (C): runtime backpressure ---
            # Wait before dispatching a new subprocess if free RAM is low.
            # Each worker thread hits this independently, so N workers
            # waiting for memory all proceed as soon as the condition
            # clears (no central coordinator needed).
            try:
                mem_safety.wait_for_available_memory(
                    min_free_gb=args.mem_backpressure_gb,
                )
            except TimeoutError as e:
                print(f"  [{idx+1}/{len(jobs)}] {grp:>6s} {name:<20s} "
                      f"BACKPRESSURE_TIMEOUT: {e}",
                      file=sys.stderr, flush=True)
                return idx, (grp, name, spec.kind, [], 0.0)

            ss = args.rnd_samples if spec.kind == "rnd" else 1
            t0 = time.time()
            recs = s3.evaluate_extension(
                spec, grp,
                t=args.t, sample_size=ss, cache=shared_cache,
                timeout_s=args.timeout, verbose=False,
                in_process=args.in_process,
                shard_workers=args.shard_workers,
                rss_cap_gb=rss_cap_gb,
            )
            dt = time.time() - t0
            return idx, (grp, name, spec.kind, recs, dt)

        workers = 1 if args.in_process else max(1, args.workers)
        if workers == 1:
            for idx_job in enumerate(jobs):
                idx, out = _run_one(idx_job)
                results[idx] = out
                grp, name, _, recs, dt = out
                best = min(r.delta for r in recs) if recs else float("nan")
                print(f"  [{idx+1}/{len(jobs)}] {grp:>6s} {name:<20s} "
                      f"best_delta={best:.4f}  {dt:.1f}s",
                      file=sys.stderr, flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_run_one, ij) for ij in enumerate(jobs)]
                done = 0
                for fut in as_completed(futs):
                    idx, out = fut.result()
                    results[idx] = out
                    done += 1
                    grp, name, _, recs, dt = out
                    best = min(r.delta for r in recs) if recs else float("nan")
                    print(f"  [{done}/{len(jobs)}] {grp:>6s} {name:<20s} "
                          f"best_delta={best:.4f}  {dt:.1f}s",
                          file=sys.stderr, flush=True)

    # Ranked table: best δ + mean ± std + Q_T.
    from math import sqrt
    from swiftbot.tools import qco as qcomod
    rows = []
    for grp, name, kind, recs, dt in results:
        group_spec = gmod.REGISTRY[grp]
        q_opt = qcomod.q_opt(group_spec.expected_size)
        deltas = [r.delta for r in recs]
        n = len(deltas)
        if n:
            best = min(deltas)
            mean = sum(deltas) / n
            std = sqrt(sum((d - mean) ** 2 for d in deltas) / n) if n > 1 else 0.0
        else:
            best = mean = std = float("nan")
        qt_best = qcomod.compute_qt(best, group_spec.expected_size)
        qt_mean = qcomod.compute_qt(mean, group_spec.expected_size)
        rows.append({
            "group": grp, "name": name, "kind": kind, "n_samples": n,
            "best_delta": best, "mean_delta": mean, "std_delta": std,
            "qt_best": qt_best, "qt_mean": qt_mean,
            "q_opt": q_opt, "seconds": dt,
        })

    rows.sort(key=lambda r: (r["group"], r["best_delta"]))
    print()
    print(f"{'group':>6s}  {'extension':<22s}  {'kind':<13s}  "
          f"{'n':>3s}  {'best_δ':>8s}  {'mean_δ':>8s}  {'std_δ':>7s}  "
          f"{'Q_T(b)':>7s}  {'Q_T(μ)':>7s}  {'Q_opt':>7s}  {'sec':>6s}")
    print("-" * 110)
    for r in rows:
        qb = f"{r['qt_best']:7.3f}" if r['qt_best'] == r['qt_best'] else "    nan"
        qm = f"{r['qt_mean']:7.3f}" if r['qt_mean'] == r['qt_mean'] else "    nan"
        print(f"{r['group']:>6s}  {r['name']:<22s}  {r['kind']:<13s}  "
              f"{r['n_samples']:>3d}  {r['best_delta']:8.4f}  "
              f"{r['mean_delta']:8.4f}  {r['std_delta']:7.4f}  "
              f"{qb}  {qm}  {r['q_opt']:7.3f}  {r['seconds']:6.1f}")

    import json as _json
    print(_json.dumps(rows, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "explore":
        if (rc := _check_api_key("explore")) is not None:
            return rc
        cache_kwargs: dict = {"run_id": args.run_id}
        if args.db is not None:
            cache_kwargs["path"] = args.db
        with Cache(**cache_kwargs) as cache:
            llm = AnthropicLLM(model=args.model)
            result = run(args.dim, cache=cache, llm=llm, top_n=args.top_n,
                         run_id=args.run_id)
        print(result.model_dump_json(indent=2))
        return 0

    if args.cmd == "sweep":
        if (rc := _check_api_key("sweep")) is not None:
            return rc
        cache_kwargs = {"run_id": args.run_id}
        if args.db is not None:
            cache_kwargs["path"] = args.db
        with Cache(**cache_kwargs) as cache:
            llm = AnthropicLLM(model=args.model)
            result = sweep(
                args.dim,
                t=args.t,
                sample_size=args.samples,
                cache=cache,
                llm=llm,
                top_n=args.top_n,
                max_extensions_per_group=args.max_per_group,
                timeout_s=args.timeout,
                run_id=args.run_id,
                verbose=not args.quiet,
                include_coverage=args.with_coverage,
                coverage_bases=tuple(args.coverage_bases.split(",")),
                coverage_max_depth=args.coverage_depth,
                coverage_n_parametric=args.coverage_n_parametric,
                workers=args.workers,
            )
        # Summary table to stderr, full JSON to stdout (pipe-friendly).
        print(format_sweep_table(result), file=sys.stderr)
        print(result.model_dump_json(indent=2))
        return 0

    if args.cmd == "cover-panel":
        import numpy as np
        from swiftbot.stages.target_coverage import evaluate_coverage_by_name
        from swiftbot.targets import get_target_family

        def _rz(phi: float):
            return np.array(
                [[np.exp(-1j * phi / 2), 0.0],
                 [0.0, np.exp(1j * phi / 2)]],
                dtype=complex,
            )

        def _norm_sud(M):
            d = M.shape[0]
            det = np.linalg.det(M)
            return M / det ** (1.0 / d)

        # Curated d=2 panel — mirrors the Scope-A paper-validation experiments.
        if args.panel == "lamm_d2":
            import math
            panel = [
                ("P(π/4) canonical",  _rz(math.pi / 4),
                 ExtensionSpec(kind="angle", params={"theta": math.pi / 4}, rationale="panel")),
                ("P(π/8)",            _rz(math.pi / 8),
                 ExtensionSpec(kind="angle", params={"theta": math.pi / 8}, rationale="panel")),
                ("P(2π/9)",           _rz(2 * math.pi / 9),
                 ExtensionSpec(kind="angle", params={"theta": 2 * math.pi / 9}, rationale="panel")),
                ("P(π/9)",            _rz(math.pi / 9),
                 ExtensionSpec(kind="angle", params={"theta": math.pi / 9}, rationale="panel")),
                ("P(π/18)",           _rz(math.pi / 18),
                 ExtensionSpec(kind="angle", params={"theta": math.pi / 18}, rationale="panel")),
                ("T_24 super-golden (Clifford)",
                 _norm_sud(np.array(
                     [[-1 - np.sqrt(2), 2 - np.sqrt(2) + 1j],
                      [2 - np.sqrt(2) - 1j, 1 + np.sqrt(2)]], dtype=complex)),
                 ExtensionSpec(kind="mat",
                               params={"matrix": "super_golden_T24"}, rationale="panel")),
                ("T_12 super-golden (Hurwitz)",
                 _norm_sud(np.array(
                     [[3, 1 - 1j], [1 + 1j, -3]], dtype=complex)),
                 ExtensionSpec(kind="mat",
                               params={"matrix": "super_golden_T12"}, rationale="panel")),
            ]

        cache_kwargs: dict = {"run_id": args.run_id}
        if args.db is not None:
            cache_kwargs["path"] = args.db

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        records: list[tuple[str, object]] = [(None, None)] * len(panel)  # preserve order
        progress_lock = threading.Lock()
        done = [0]
        total = len(panel)

        def _run_panel_entry(idx_name_mat_spec):
            idx, (name, T_matrix, spec) = idx_name_mat_spec
            fp = extension_fingerprint(spec)
            rec = evaluate_coverage_by_name(
                args.base, T_matrix, args.family,
                max_depth=args.max_depth,
                n_parametric_samples=args.n_parametric,
                eps_hit=args.eps,
                max_unique=args.max_unique,
                cache=cache,
                ext_fingerprint=fp,
            )
            records[idx] = (name, rec)
            with progress_lock:
                done[0] += 1
                print(
                    f"  [{done[0]}/{total}] {name}  mean={rec.mean_dist:.4f}  "
                    f"hits={rec.hits_count}/{rec.n_targets}  "
                    f"fingerprint={fp[:10]}…",
                    file=sys.stderr, flush=True,
                )

        with Cache(**cache_kwargs) as cache:
            items = list(enumerate(panel))
            if args.workers <= 1:
                for item in items:
                    _run_panel_entry(item)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for _ in as_completed([pool.submit(_run_panel_entry, it) for it in items]):
                        pass

        # Ranked table (sorted by mean_dist ascending).
        records_sorted = sorted(records, key=lambda r: r[1].mean_dist)
        print("", file=sys.stderr)
        print(f"Cover panel  family={args.family}  base={args.base}  "
              f"depth≤{args.max_depth}  n_theta={args.n_parametric}",
              file=sys.stderr)
        print(f"{'extension':<32}  {'visited':>8}  {'mean_d':>8}  {'max_d':>8}  "
              f"{'hits':>9}  {'⟨T⟩_hit':>9}", file=sys.stderr)
        print("-" * 86, file=sys.stderr)
        for name, rec in records_sorted:
            tc = f"{rec.mean_t_count_hits:.2f}" if rec.mean_t_count_hits is not None else "  —  "
            print(
                f"{name:<32}  {rec.visited:>8d}  {rec.mean_dist:>8.4f}  "
                f"{rec.max_dist:>8.4f}  "
                f"{rec.hits_count:>4}/{rec.n_targets:<4}  {tc:>9}",
                file=sys.stderr,
            )

        # JSON array of records to stdout.
        import json
        out = [
            {"panel_name": name, "record": rec.model_dump()}
            for name, rec in records_sorted
        ]
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "cover":
        if args.list:
            for fam in list_target_families():
                n_disc = len(fam.discrete)
                para = "yes" if fam.parametric is not None else "no"
                print(f"  {fam.name:<20} d={fam.qudit_dim}  discrete={n_disc:<3}  "
                      f"parametric={para}  — {fam.description[:80]}")
            return 0

        # Build ExtensionSpec from CLI args
        params: dict = {}
        if args.ext_kind == "angle":
            if args.ext_theta is None:
                raise SystemExit("--ext-kind angle requires --ext-theta <radians>")
            params["theta"] = args.ext_theta
        elif args.ext_kind == "angles":
            if args.ext_phases is None:
                raise SystemExit("--ext-kind angles requires --ext-phases")
            params["phases"] = [float(x) for x in args.ext_phases.split(",")]
        elif args.ext_kind == "howard_vala":
            if args.ext_hv is None:
                raise SystemExit("--ext-kind howard_vala requires --ext-hv z,g,e")
            z, g, e = [int(x) for x in args.ext_hv.split(",")]
            params = {"z": z, "gamma": g, "eps": e}
        elif args.ext_kind == "mat":
            if args.ext_mat_npy is None:
                raise SystemExit("--ext-kind mat requires --ext-mat-npy PATH")
            import numpy as np
            M = np.load(args.ext_mat_npy)
            params["matrix"] = M.tolist()

        spec = ExtensionSpec(kind=args.ext_kind, params=params, rationale="cli")
        fp = extension_fingerprint(spec)
        # Resolve the dimension of the extension from the target family
        fam = next((f for f in list_target_families() if f.name == args.family), None)
        if fam is None:
            raise SystemExit(f"unknown family: {args.family}")
        T_matrix = materialize_extension(spec, fam.qudit_dim)
        if T_matrix is None:
            raise SystemExit("ext_kind 'rnd' is not a concrete target for cover; "
                             "pick angle/angles/mat/howard_vala.")

        cache_kwargs: dict = {"run_id": args.run_id}
        if args.db is not None:
            cache_kwargs["path"] = args.db
        with Cache(**cache_kwargs) as cache:
            record = evaluate_coverage_by_name(
                args.base, T_matrix, args.family,
                max_depth=args.max_depth,
                n_parametric_samples=args.n_parametric,
                eps_hit=args.eps,
                max_unique=args.max_unique,
                cache=cache,
                ext_fingerprint=fp,
            )
        # Emit structured JSON (full record) to stdout; summary to stderr.
        print(
            f"cover: base={args.base}  family={args.family}  ext_kind={spec.kind}  "
            f"visited={record.visited}  mean_dist={record.mean_dist:.4f}  "
            f"max_dist={record.max_dist:.4f}  hits={record.hits_count}/{record.n_targets}",
            file=sys.stderr,
        )
        print(record.model_dump_json(indent=2))
        return 0

    if args.cmd == "q-panel":
        return _run_q_panel(args)

    if args.cmd == "q-panel-summary":
        return _run_q_panel_summary(args)

    if args.cmd == "codes":
        if args.group is not None:
            hits, note = codesmod.codes_for_group(args.group)
        elif args.dim is not None:
            hits = codesmod.codes_for_dim(args.dim)
            note = f"{len(hits)} curated code(s) with qudit_dim={args.dim}."
        else:
            hits = codesmod.list_all_codes()
            note = f"{len(hits)} curated code(s) in catalog."
        print(note, file=sys.stderr)
        for c in hits:
            params = (
                f"[[{c.n},{c.k},{c.distance}]]_{c.qudit_dim}"
                if c.n is not None and c.k is not None and c.distance is not None
                else f"family (qudit_dim={c.qudit_dim})"
            )
            tg = ", ".join(c.transversal_groups) or "—"
            suffix = "  (research needed)" if c.research_needed else ""
            print(f"  {c.name}  {params}  transversal: {tg}{suffix}")
            if c.reference:
                print(f"      {c.reference}")
        if args.distillation:
            print("\nDistillation protocols:", file=sys.stderr)
            for p_ in distmod.list_all_protocols():
                yp = f"γ={p_.yield_parameter}" if p_.yield_parameter is not None else "γ=—"
                print(f"  {p_.protocol_name}  →  family={p_.target_gate_family}, "
                      f"d={p_.qudit_dim}, code={p_.code_name}, {yp}")
                if p_.reference:
                    print(f"      {p_.reference}")
        return 0

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
