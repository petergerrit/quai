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

        records = []
        with Cache(**cache_kwargs) as cache:
            for name, T_matrix, spec in panel:
                fp = extension_fingerprint(spec)
                print(f"  running: {name}  (fingerprint={fp[:10]}…)", file=sys.stderr, flush=True)
                rec = evaluate_coverage_by_name(
                    args.base, T_matrix, args.family,
                    max_depth=args.max_depth,
                    n_parametric_samples=args.n_parametric,
                    eps_hit=args.eps,
                    max_unique=args.max_unique,
                    cache=cache,
                    ext_fingerprint=fp,
                )
                records.append((name, rec))

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
