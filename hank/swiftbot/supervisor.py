"""SWIFTbot supervisor: a state machine with LLM judgment at stage boundaries.

Control flow for `run(dim=...)`:

    Stage 1 — explore groups
        deterministic: list registry, for each compute Sawicki (cache-backed).
        LLM: rank groups by suitability as a universal-extension base.

    Stage 2 — propose extensions
        deterministic: assemble context (group metadata + Sawicki verdict +
                       any cached Q_T results).
        LLM: propose a short list of extension specs to evaluate next.

    (Stage 3 — actually running Q_T on the proposed extensions lives in
     `stages/s3_efficiency.py` once the generator + job-runner are wired up;
     for now the supervisor returns the proposals without executing them.)

Outputs: everything the supervisor touches goes into the SQLite cache with
provenance (git SHA, hostname, run_id), so you can pick up from a previous
run or query results later.
"""
from __future__ import annotations

import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from swiftbot.kb.cache import Cache, matrices_key
from swiftbot.llm import AnthropicLLM, LLMBackend
from swiftbot.state import (
    CodeRecord,
    CoverageRecord,
    DistillationRecord,
    QTRecord,
    SawickiRecord,
)
from swiftbot.tools import codes as codesmod
from swiftbot.tools import distillation as distmod
from swiftbot.tools import groups as gmod
from swiftbot.tools import sawicki as sawmod
from swiftbot.targets import list_target_families


# ---------------------------------------------------------------------------
# Structured-output models for the LLM
# ---------------------------------------------------------------------------

class GroupPriority(BaseModel):
    """One pick from the LLM: a registered group name + why it was chosen."""
    name: str = Field(..., description="Registered group name, e.g. 'BI' or 'S216'.")
    rationale: str = Field(..., description="One-sentence justification.")


class GroupSelection(BaseModel):
    """Ordered list of registered groups the supervisor should study next."""
    selections: list[GroupPriority] = Field(..., min_length=1, max_length=8)


ExtKind = Literal["angle", "angles", "howard_vala", "mat", "rnd"]


class ExtensionSpec(BaseModel):
    """A single candidate extension T to adjoin to a chosen base group C."""
    kind: ExtKind = Field(
        ...,
        description=(
            "Extension family. 'angle': diagonal P(θ). 'angles': d-1 diagonal "
            "phases. 'howard_vala': three-parameter qudit T-gate family. "
            "'mat': explicit SU(d) matrix. 'rnd': Haar-random completion."
        ),
    )
    params: dict = Field(
        default_factory=dict,
        description="Free-form parameters scoped to `kind` (angles, indices, etc.).",
    )
    rationale: str = Field(..., description="Why this extension is worth trying.")


class ExtensionProposal(BaseModel):
    """LLM's list of extension candidates for one base group."""
    extensions: list[ExtensionSpec] = Field(..., min_length=1, max_length=6)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = """\
You are SWIFTbot, a research-grade agent that ranks finite subgroups of SU(d)
as candidate base groups C for universal quantum gate set extensions.

Context for your ranking:
- Necessary condition for ⟨C, T⟩ universality: the adjoint rep of ⟨C, T⟩ on
  su(d) must be irreducible. If C alone already has reducible Ad, any
  extension T must break the invariant subspaces — usually a hard ask.
- Paper convention: Q_T = log|C| / log(1/δ(ν_S,t)); Kesten-McKay lower bound
  is ~ 2 for large |C|. Larger |C| gives more conjugates and typically a
  smaller δ, but raises implementation/calibration cost.
- Fault-tolerance angle: prefer groups that are known to be realisable as
  transversal gate sets of some quantum error-correcting code. SWIFTbot's
  Stage-4 catalog lists concretely-known codes; the stage-1 user message
  annotates each group with its count. Groups with 'codes=0' are not ruled
  out, but their downstream fault-tolerance story is speculative.
- Dimension-specific knowledge:
    * d=2: Clifford/Hurwitz are the canonical non-trivial bases (Super-Golden
      literature). BI/BO/BT are the SU(2) lifts of icosahedral/octahedral/
      tetrahedral.
    * d=3: qutrit Clifford+T is arithmetically "thin" (Slowik et al.,
      Yard et al.); prefer Σ(72×3) (S216) or Σ(216×3) (S648) as bases when
      an irreducible Ad is required.
    * d=4: s7f (5040 elements) is the only registered group with
      irreducible Ad; smaller registered d=4 groups are reducible.

Always give at least one pick; call the tool exactly once with your
ranked list.
"""

_STAGE2_SYSTEM = """\
You are SWIFTbot, proposing extensions for a chosen base group C ⊂ SU(d).

Extension families available in the pipeline:
  angle         — T = diag(1, e^{iθ}, ..., 1); one angle θ ∈ (0, 2π).
  angles        — T diagonal with d-1 phases summing so det(T)=1.
  howard_vala   — (z', γ', ε) three-parameter diagonal π/8 qudit family
                  (Howard & Vala 2012, prime d). Includes Campbell M(p).
                  FULLY IMPLEMENTED for prime d (2, 3, 5, 7, ...).
  mat           — an explicit d×d SU(d) matrix (non-diagonal).
  rnd           — Haar-random gate of optional finite order r.

Guidance:
- If C has reducible Ad (commutant_dim > 1), favour 'mat' or 'rnd'
  proposals: only non-diagonal extensions can break the invariant
  subspaces of the base group.
- If C has irreducible Ad, diagonal extensions are meaningful — sweep
  'angle' or 'howard_vala' with a handful of representative parameter
  choices.
- For qutrits: prefer howard_vala with (z'=1, γ'=1, ε=0) (Campbell) and
  small perturbations. Include at least one non-diagonal 'mat' or 'rnd'
  control.
- Fault-tolerance angle: SWIFTbot's Stage-5 catalog maps some extension
  families to known magic-state distillation protocols. The canonical
  qubit T (angle θ = π/4) maps to the Bravyi-Kitaev 15-to-1 via Reed-Muller;
  Howard-Vala for d=3 maps to the [[9m-k,k,2]]_3 qutrit triorthogonal
  family. Proposing at least one extension in a family with a known
  distillation protocol gives the paper a direct fault-tolerance story.
- Limit to 3-6 candidates per call; they'll be evaluated one at a time.

Always call the tool exactly once with your proposal list.
"""


# ---------------------------------------------------------------------------
# Stage 1 — explore groups
# ---------------------------------------------------------------------------

def _ensure_sawicki(cache: Cache, group_name: str, matrices: np.ndarray) -> tuple[str, SawickiRecord]:
    """Put a group into the cache and make sure it has a Sawicki record."""
    key = cache.put_group(
        matrices,
        name=group_name,
        source=f"registry:{group_name}",
        projective=gmod.REGISTRY[group_name].projective,
    )
    rec = cache.get_sawicki(key)
    if rec is not None:
        return key, rec
    result = sawmod.check_universality(list(matrices))
    rec = SawickiRecord(
        target_key=key,
        verdict=result.verdict,
        commutant_dim=result.commutant_dim,
        irreducible=result.irreducible,
        min_distance_to_center=result.min_distance_to_center,
        has_near_center_element=result.has_near_center_element,
        notes=result.notes,
    )
    cache.put_sawicki(rec)
    return key, rec


def stage1_explore_groups(
    dim: int,
    *,
    cache: Cache,
    llm: LLMBackend,
) -> tuple[GroupSelection, dict[str, SawickiRecord]]:
    """Run Sawicki on every registered group with given dim; LLM ranks them."""
    specs = gmod.list_groups(d=dim)
    if not specs:
        raise ValueError(f"no groups registered for dim={dim}")

    verdicts: dict[str, SawickiRecord] = {}
    for spec in specs:
        mats = np.asarray(gmod.get_group(spec.name))
        _, rec = _ensure_sawicki(cache, spec.name, mats)
        verdicts[spec.name] = rec

    lines: list[str] = []
    for spec in specs:
        v = verdicts[spec.name]
        code_hits, _ = codesmod.codes_for_group(spec.name)
        lines.append(
            f"- {spec.name}: |C|={spec.expected_size}, "
            f"commutant_dim={v.commutant_dim}, irreducible={v.irreducible}, "
            f"verdict={v.verdict}, "
            f"min_dist_centre={v.min_distance_to_center:.3f}, "
            f"codes={len(code_hits)}"
            + (f"  — {spec.notes}" if spec.notes else "")
        )
    user = (
        f"Target qudit dimension: d = {dim}\n\n"
        f"Registered finite subgroups of SU({dim}) with their Sawicki status "
        f"(already computed):\n" + "\n".join(lines) + "\n\n"
        "Rank the groups you want SWIFTbot to explore further, most-promising first."
    )
    selection = llm.ask_structured(
        system=_STAGE1_SYSTEM,
        user=user,
        output_model=GroupSelection,
        tool_name="rank_groups",
    )
    registered = {spec.name for spec in specs}
    for pick in selection.selections:
        if pick.name not in registered:
            raise ValueError(
                f"LLM picked unregistered group {pick.name!r}. "
                f"Registered for d={dim}: {sorted(registered)}"
            )
    return selection, verdicts


# ---------------------------------------------------------------------------
# Stage 2 — propose extensions
# ---------------------------------------------------------------------------

def stage2_propose_extensions(
    group_name: str,
    verdict: SawickiRecord,
    *,
    cache: Cache,
    llm: LLMBackend,
) -> ExtensionProposal:
    """Ask the LLM for a short list of concrete extensions to try next."""
    spec = gmod.REGISTRY[group_name]
    qt_context = ""
    prior = cache.list_qt(verdict.target_key)
    if prior:
        best = min(prior, key=lambda r: (r.qt if r.qt is not None else float("inf")))
        qt_context = (
            f"\nPrior Q_T measurements for this group (cached): {len(prior)} rows; "
            f"best Q_T = {best.qt!r} at t={best.t}, sample_id={best.sample_id}."
        )
    user = (
        f"Base group: {group_name}  (|C|={spec.expected_size}, d={spec.d})\n"
        f"Sawicki verdict: {verdict.verdict}; "
        f"commutant_dim={verdict.commutant_dim}; "
        f"irreducible={verdict.irreducible}; "
        f"min distance to centre = {verdict.min_distance_to_center:.3f}.\n"
        f"Group notes: {spec.notes or '(none)'}{qt_context}\n\n"
        "Propose 3–6 concrete extension candidates for SWIFTbot to evaluate next."
    )
    return llm.ask_structured(
        system=_STAGE2_SYSTEM,
        user=user,
        output_model=ExtensionProposal,
        tool_name="propose_extensions",
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class SupervisorResult(BaseModel):
    dim: int
    selection: GroupSelection
    verdicts: dict[str, SawickiRecord]
    proposals: dict[str, ExtensionProposal]


class SweepEvaluation(BaseModel):
    """Outcome of running Stage 3 (and 4/5) for one (base_group, extension) pair.

    `codes_found` is filled from the Stage-4 curated catalog using the base
    group's name; `distillation_found` is from Stage-5 catalog given the
    extension spec's kind/params/dimension. `codes_note` and `distillation_note`
    are human-readable strings covering the "research may be needed" case.
    """
    group_name: str
    ext_kind: str
    ext_params: dict = Field(default_factory=dict)
    ext_rationale: str = ""
    qt_records: list[QTRecord] = Field(default_factory=list)
    best_qt: float | None = None
    best_delta: float | None = None
    error: str | None = None                       # set iff Stage 3 failed
    codes_found: list[CodeRecord] = Field(default_factory=list)
    codes_note: str = ""
    distillation_found: list[DistillationRecord] = Field(default_factory=list)
    distillation_note: str = ""
    coverage_records: list[CoverageRecord] = Field(default_factory=list)
    coverage_note: str = ""


class SweepResult(BaseModel):
    """Full explore + evaluate pipeline output."""
    dim: int
    t: int
    sample_size: int
    selection: GroupSelection
    verdicts: dict[str, SawickiRecord]
    proposals: dict[str, ExtensionProposal]
    evaluations: list[SweepEvaluation]


def _progress(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr, flush=True)


def run(
    dim: int,
    *,
    cache: Cache | None = None,
    llm: LLMBackend | None = None,
    top_n: int = 3,
    run_id: str | None = None,
) -> SupervisorResult:
    """Execute Stages 1-2 for a given qudit dimension.

    Deterministic numerics (Stage 1 Sawicki on every registered group)
    always run. The LLM is called twice per top-N group: once to rank, once
    to propose extensions. The cache ensures no redundant work across runs.
    """
    owned_cache = cache is None
    cache = cache or Cache(run_id=run_id)
    llm = llm or AnthropicLLM()
    try:
        selection, verdicts = stage1_explore_groups(dim, cache=cache, llm=llm)
        proposals: dict[str, ExtensionProposal] = {}
        for priority in selection.selections[:top_n]:
            proposals[priority.name] = stage2_propose_extensions(
                priority.name,
                verdicts[priority.name],
                cache=cache,
                llm=llm,
            )
        return SupervisorResult(
            dim=dim,
            selection=selection,
            verdicts=verdicts,
            proposals=proposals,
        )
    finally:
        if owned_cache:
            cache.close()


# ---------------------------------------------------------------------------
# Stage 1 + 2 + 3 — full sweep (explore → propose → evaluate)
# ---------------------------------------------------------------------------

def sweep(
    dim: int,
    *,
    t: int,
    sample_size: int = 1,
    cache: Cache | None = None,
    llm: LLMBackend | None = None,
    top_n: int = 3,
    max_extensions_per_group: int | None = None,
    timeout_s: float | None = 600,
    run_id: str | None = None,
    verbose: bool = False,
    include_coverage: bool = False,
    coverage_bases: tuple[str, ...] = ("clifford",),
    coverage_max_depth: int = 8,
    coverage_n_parametric: int = 5,
    workers: int = 1,
) -> SweepResult:
    """Full pipeline: explore groups (stage 1+2) then actually execute every
    proposed extension via qco-main_opt (stage 3).

    Execution continues past individual failures — if one (group, extension)
    evaluation errors (e.g. howard_vala not implemented yet, or a qco timeout),
    its SweepEvaluation has `error` set and the sweep proceeds. This way a
    whole batch isn't lost to a single bad proposal.

    `workers`: if > 1, evaluate the (group, extension) stage-3 tasks in
    parallel using a ThreadPoolExecutor. Each qco subprocess releases the
    GIL while waiting on I/O, so threads give near-full speedup. SQLite
    WAL mode serialises writes to the cache, so there are no data races.
    The returned `evaluations` list preserves submission order regardless
    of worker count. Default workers=1 preserves the deterministic serial
    path used in existing tests.
    """
    # Deferred import so stage-3 stays optional for tests that don't need it.
    from swiftbot.stages import s3_efficiency as s3

    owned_cache = cache is None
    cache = cache or Cache(run_id=run_id)
    llm = llm or AnthropicLLM()

    try:
        _progress(
            f"[sweep] dim={dim}  t={t}  sample_size={sample_size}  workers={workers}",
            verbose=verbose,
        )
        explore = run(dim, cache=cache, llm=llm, top_n=top_n, run_id=run_id)

        # Stage 4/5 lookups are deterministic and cheap; compute once per group.
        group_codes: dict[str, tuple[list[CodeRecord], str]] = {}
        for group_name in explore.proposals:
            group_codes[group_name] = codesmod.codes_for_group(group_name)

        # Build an ordered list of work items. Each item is a (stable_index,
        # group_name, ExtensionSpec) tuple so downstream code can place the
        # SweepEvaluation at its original index regardless of completion order.
        work_items: list[tuple[int, str, "ExtensionSpec"]] = []
        for group_name, proposal in explore.proposals.items():
            exts = proposal.extensions
            if max_extensions_per_group is not None:
                exts = exts[:max_extensions_per_group]
            for ext in exts:
                work_items.append((len(work_items), group_name, ext))
        total = len(work_items)

        evaluations: list[SweepEvaluation | None] = [None] * total
        progress_lock = threading.Lock()
        done_counter = [0]

        def _worker(item: tuple[int, str, "ExtensionSpec"]) -> None:
            idx, group_name, ext = item
            codes_hits, codes_note = group_codes[group_name]
            qudit_dim = gmod.REGISTRY[group_name].d
            dist_hits, dist_note = distmod.protocols_for_extension(
                ext.kind, ext.params, qudit_dim,
            )
            try:
                records = s3.evaluate_extension(
                    ext, group_name,
                    t=t, sample_size=sample_size,
                    cache=cache,
                    timeout_s=timeout_s,
                    verbose=False,
                )
                qts = [r.qt for r in records if r.qt is not None]
                best_qt = min(qts) if qts else None
                deltas = [r.delta for r in records]
                best_delta = min(deltas) if deltas else None
                cov_records, cov_note = _maybe_run_coverage(
                    ext, qudit_dim, coverage_bases, cache,
                    enabled=include_coverage,
                    max_depth=coverage_max_depth,
                    n_parametric=coverage_n_parametric,
                )
                evaluations[idx] = SweepEvaluation(
                    group_name=group_name,
                    ext_kind=ext.kind,
                    ext_params=ext.params,
                    ext_rationale=ext.rationale,
                    qt_records=records,
                    best_qt=best_qt,
                    best_delta=best_delta,
                    codes_found=codes_hits,
                    codes_note=codes_note,
                    distillation_found=dist_hits,
                    distillation_note=dist_note,
                    coverage_records=cov_records,
                    coverage_note=cov_note,
                )
                with progress_lock:
                    done_counter[0] += 1
                    cov_summary = (
                        f", coverage={len(cov_records)} run(s)"
                        if include_coverage else ""
                    )
                    _progress(
                        f"  [{done_counter[0]}/{total}] {group_name} + "
                        f"{ext.kind}({ext.params})  "
                        f"→ Q_T={best_qt!r}, δ={best_delta!r}, "
                        f"codes={len(codes_hits)}, distillations={len(dist_hits)}"
                        f"{cov_summary}",
                        verbose=verbose,
                    )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                evaluations[idx] = SweepEvaluation(
                    group_name=group_name,
                    ext_kind=ext.kind,
                    ext_params=ext.params,
                    ext_rationale=ext.rationale,
                    error=msg,
                    codes_found=codes_hits,
                    codes_note=codes_note,
                    distillation_found=dist_hits,
                    distillation_note=dist_note,
                )
                with progress_lock:
                    done_counter[0] += 1
                    _progress(
                        f"  [{done_counter[0]}/{total}] {group_name} + "
                        f"{ext.kind}({ext.params})  ✗ {msg}",
                        verbose=verbose,
                    )

        if workers <= 1:
            for item in work_items:
                _worker(item)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in as_completed([pool.submit(_worker, item) for item in work_items]):
                    pass

        # By construction every slot has been filled; narrow the type.
        final_evaluations: list[SweepEvaluation] = [e for e in evaluations if e is not None]
        assert len(final_evaluations) == total, "worker dropped an evaluation"

        return SweepResult(
            dim=dim,
            t=t,
            sample_size=sample_size,
            selection=explore.selection,
            verdicts=explore.verdicts,
            proposals=explore.proposals,
            evaluations=final_evaluations,
        )
    finally:
        if owned_cache:
            cache.close()


def _maybe_run_coverage(
    ext: "ExtensionSpec",
    qudit_dim: int,
    coverage_bases: tuple[str, ...],
    cache: Cache,
    *,
    enabled: bool,
    max_depth: int,
    n_parametric: int,
) -> tuple[list[CoverageRecord], str]:
    """If `enabled`, run evaluate_coverage_by_name for each base in
    coverage_bases against every target family registered at qudit_dim.
    Returns (records, note). Records may be empty if no matching families."""
    if not enabled:
        return [], "include_coverage=False; skipped"

    # Local imports to avoid circular references + keep sweep light when unused.
    from swiftbot.stages.s3_efficiency import extension_fingerprint, materialize_extension
    from swiftbot.stages.target_coverage import evaluate_coverage_by_name

    try:
        T_matrix = materialize_extension(ext, qudit_dim)
    except Exception:
        return [], f"materialize_extension failed (kind={ext.kind})"
    if T_matrix is None:
        return [], "ext kind 'rnd' has no concrete matrix; coverage skipped"

    fp = extension_fingerprint(ext)
    families = [f for f in list_target_families() if f.qudit_dim == qudit_dim]
    if not families:
        return [], f"no target families registered for dim={qudit_dim}"

    records: list[CoverageRecord] = []
    for base_name in coverage_bases:
        for fam in families:
            try:
                rec = evaluate_coverage_by_name(
                    base_name, T_matrix, fam.name,
                    max_depth=max_depth,
                    n_parametric_samples=n_parametric,
                    cache=cache,
                    ext_fingerprint=fp,
                )
                records.append(rec)
            except Exception:
                # Non-fatal: skip this (base, family) but continue the sweep.
                pass
    note = (
        f"coverage: {len(records)} run(s) over {len(coverage_bases)} base(s) × "
        f"{len(families)} family(ies) at dim={qudit_dim}, depth≤{max_depth}"
    )
    return records, note


def format_sweep_table(result: SweepResult) -> str:
    """Human-readable ranked table of sweep evaluations for stderr output."""
    rows = sorted(
        result.evaluations,
        key=lambda e: (e.best_qt if e.best_qt is not None else math.inf),
    )
    lines: list[str] = []
    header = (
        f"Sweep summary  (dim={result.dim}, t={result.t}, samples={result.sample_size})"
    )
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(
        f"{'group':<12} {'ext_kind':<12} {'best Q_T':>10}  {'best δ':>8}  "
        f"{'codes':>5}  {'dist':>4}  {'cov_best_μd':>11}  {'cov_hits':>9}  status"
    )
    for e in rows:
        status = "ok" if e.error is None else f"ERR: {e.error}"
        qt_s  = f"{e.best_qt:.4f}" if e.best_qt  is not None else "    —"
        dl_s  = f"{e.best_delta:.4f}" if e.best_delta is not None else "    —"
        if e.coverage_records:
            best_cov = min(e.coverage_records, key=lambda r: r.mean_dist)
            cov_s = f"{best_cov.mean_dist:.4f}"
            hits_s = f"{best_cov.hits_count}/{best_cov.n_targets}"
        else:
            cov_s = "      —"
            hits_s = "    —"
        lines.append(
            f"{e.group_name:<12} {e.ext_kind:<12} {qt_s:>10}  {dl_s:>8}  "
            f"{len(e.codes_found):>5}  {len(e.distillation_found):>4}  "
            f"{cov_s:>11}  {hits_s:>9}  {status}"
        )
    return "\n".join(lines)
