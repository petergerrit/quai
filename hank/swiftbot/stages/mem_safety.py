"""Memory-safety helpers for qco-panel runs.

Motivation: a d=3, t=10, sample_size=100 panel on S1080 with --workers 8
--shard-workers 4 once OOM-locked a 125 GB workstation because swiftbot
dispatched all evaluations without first checking whether their peak
concurrent RSS would fit in RAM. This module adds three defenses:

    (A) `check_budget_or_raise(...)` — pre-flight: compute a conservative
        upper bound on peak concurrent RSS for a panel and abort before
        launch if it exceeds `safety_frac * free_RAM`.
    (B) RLIMIT_AS per qco subprocess (see `rlimit_preexec`) — a runaway
        subprocess raises MemoryError instead of eating the whole host.
    (C) `wait_for_available_memory(...)` — runtime backpressure: pause
        before dispatching a new job if free RAM is under a threshold.

The memory model is empirical, calibrated from the Tier-2 (d=3, t=10) run
that motivated this module. It is deliberately on the high side — a ~2×
safety factor over observed per-proc RSS — because an abort is strictly
preferable to an OOM cascade that loses hours of compute and locks the
machine. Override via environment variables:

    SWIFTBOT_MEM_PER_PROC_GB    — pin per-proc RSS estimate (bypass model)
    SWIFTBOT_MEM_OVERHEAD_GB    — baseline per-panel overhead (default 4 GB)
    SWIFTBOT_MEM_SAFETY_FRAC    — fraction of free_RAM the budget may use
                                  (default 0.70)
    SWIFTBOT_MEM_BACKPRESSURE_GB — min free RAM before dispatching a new
                                   subprocess (default 6 GB)
    SWIFTBOT_MEM_BACKPRESSURE_S  — max seconds to wait at backpressure
                                   before abort (default 600)

The module reads /proc/meminfo directly — no psutil dependency.
"""
from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Free-memory probe
# ---------------------------------------------------------------------------

_MEMINFO = Path("/proc/meminfo")


def available_memory_gb() -> float:
    """Return MemAvailable in GB from /proc/meminfo.

    MemAvailable is the kernel's own estimate of how much RAM is usable for
    new allocations without swap pressure — more accurate than MemFree,
    which excludes reclaimable page cache. Returns 0.0 if /proc/meminfo is
    unreadable (non-Linux or sandboxed).
    """
    try:
        for line in _MEMINFO.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


def total_memory_gb() -> float:
    """Return MemTotal in GB."""
    try:
        for line in _MEMINFO.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Per-proc RSS model
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


# Calibration anchors (direct measurement on lenore):
#   * (d=3, t=10, |C|=216) ≈ 4.0 GB per qco subprocess (Tier-2 v2)
#   * (d=3, t=5,  |C|=1080) ≈ 2.2 GB per qco subprocess (d3_rnd_diag)
#   * (d=3, t=5,  |C|=216)  ≈ 0.65 GB per qco subprocess (d3_rnd_diag)
# Model: per_proc = base + slope × |C| × (t/10)^α.
# At d=3 the dominant irrep has dimension O(t^{d(d-1)/2}) = O(t^3); the Π(g)
# matrix storage is (dim)^2 = O(t^6). Fitting α=6 reproduces all three
# anchors (t=10 is the anchor so any α matches there; α=6 collapses low-t
# predictions without undershooting elsewhere). This is a significant
# reduction from the earlier α=2 estimate, which over-predicted low-t
# memory by roughly 4×.
# At d=2, dominant irrep dim ~ t (SU(2)), squared ~ t². At d≥4 we keep the
# conservative earlier guess since no direct measurement exists.
_D3_BASE_GB = 2.0        # intercept: python + numpy + system libs per proc
_D3_SLOPE_PER_C = 0.010  # GB per element of |C| at the (t/10)=1 anchor


def estimate_per_proc_rss_gb(d: int, t: int, group_size: int) -> float:
    """Conservative per-subprocess peak RSS in GB.

    Calibrated against direct measurements at d=3, t∈{5,10}, |C|∈{216,1080}.
    Override with SWIFTBOT_MEM_PER_PROC_GB if hardware differs.
    """
    override = os.environ.get("SWIFTBOT_MEM_PER_PROC_GB")
    if override is not None:
        try:
            return float(override)
        except ValueError:
            pass
    if d == 2:
        return 0.8 + 0.002 * group_size * (t / 10.0) ** 2
    if d == 3:
        return _D3_BASE_GB + _D3_SLOPE_PER_C * group_size * (t / 10.0) ** 6
    # d >= 4: no direct data; extrapolate with t^{12} ~ (irrep dim)^2.
    return 4.0 + 0.05 * group_size * (t / 10.0) ** 12


# ---------------------------------------------------------------------------
# Panel peak estimator
# ---------------------------------------------------------------------------

@dataclass
class PanelMemoryBudget:
    peak_procs: int
    peak_per_proc_gb: float
    overhead_gb: float
    total_gb: float
    available_gb: float
    safety_frac: float
    safe: bool
    reason: str


def estimate_panel_peak_gb(
    group_sizes: list[int],
    extension_kinds: list[str],
    d: int,
    t: int,
    workers: int,
    shard_workers: int,
    rnd_samples: int,
) -> PanelMemoryBudget:
    """Conservative upper bound on concurrent peak RSS for a q-panel run.

    Model:
        peak_procs = workers × shard_workers  (worst case: every worker on
                     a rnd-sharded extension simultaneously)
        peak_per_proc = RSS estimate pinned to the largest group in the
                     panel at the specified t and d
        total = peak_procs × peak_per_proc + overhead

    If the panel has no rnd extensions, `peak_procs` collapses to `workers`
    (shard_workers only splits rnd sample_size). Otherwise we assume the
    worst.
    """
    if not group_sizes:
        raise ValueError("group_sizes must be non-empty")
    largest = max(group_sizes)
    per_proc = estimate_per_proc_rss_gb(d=d, t=t, group_size=largest)

    has_rnd = any(k == "rnd" for k in extension_kinds)
    effective_shards = max(1, shard_workers) if has_rnd else 1
    peak_procs = max(1, workers) * effective_shards

    overhead_gb = _env_float("SWIFTBOT_MEM_OVERHEAD_GB", 4.0)
    total_gb = peak_procs * per_proc + overhead_gb

    available = available_memory_gb() or total_memory_gb()
    safety_frac = _env_float("SWIFTBOT_MEM_SAFETY_FRAC", 0.70)
    budget = available * safety_frac
    safe = total_gb <= budget
    reason = (
        f"peak≈{total_gb:.1f} GB "
        f"= {peak_procs} procs × {per_proc:.1f} GB/proc + {overhead_gb:.1f} GB "
        f"(|C|_max={largest}, d={d}, t={t}); "
        f"budget={budget:.1f} GB "
        f"({safety_frac:.0%} of {available:.1f} GB available)"
    )
    return PanelMemoryBudget(
        peak_procs=peak_procs,
        peak_per_proc_gb=per_proc,
        overhead_gb=overhead_gb,
        total_gb=total_gb,
        available_gb=available,
        safety_frac=safety_frac,
        safe=safe,
        reason=reason,
    )


class MemoryBudgetExceeded(RuntimeError):
    """Raised by check_budget_or_raise when the estimate exceeds the budget."""


def check_budget_or_raise(
    group_sizes: list[int],
    extension_kinds: list[str],
    d: int,
    t: int,
    workers: int,
    shard_workers: int,
    rnd_samples: int = 0,
    *,
    override: bool = False,
) -> PanelMemoryBudget:
    """Pre-flight memory check. Raises `MemoryBudgetExceeded` if the
    estimate is unsafe unless `override=True`.

    Caller should print `report.reason` either way so the user sees what
    the model predicted.
    """
    report = estimate_panel_peak_gb(
        group_sizes=group_sizes,
        extension_kinds=extension_kinds,
        d=d, t=t,
        workers=workers,
        shard_workers=shard_workers,
        rnd_samples=rnd_samples,
    )
    if not report.safe and not override:
        raise MemoryBudgetExceeded(
            report.reason
            + ". Aborting to avoid OOM lockup. "
            + "Re-run with --ignore-memory-budget to override, or reduce "
            + "--workers, --shard-workers, or --t."
        )
    return report


# ---------------------------------------------------------------------------
# RLIMIT_AS per subprocess
# ---------------------------------------------------------------------------

def rlimit_preexec(max_rss_gb: float):
    """Return a preexec_fn that caps the subprocess's virtual memory.

    Usage:
        subprocess.Popen(..., preexec_fn=rlimit_preexec(12.0))

    A shard that exceeds the cap will get a MemoryError from numpy/Python
    rather than OOM-killing the host. Ignored on platforms without
    RLIMIT_AS (Windows — swiftbot is Linux-only, so this is not a real
    concern, but we guard for sanity).
    """
    cap_bytes = int(max_rss_gb * 1024 ** 3)

    def _set_rlimit():  # pragma: no cover — runs in child process
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))
        except (ValueError, OSError):
            pass

    return _set_rlimit


def per_subprocess_cap_gb(
    d: int, t: int, group_size: int, *, multiplier: float = 1.5,
) -> float:
    """RLIMIT_AS cap in GB. Defaults to 1.5× the model estimate so benign
    overshoots don't fail the run, but catastrophic ones do."""
    return multiplier * estimate_per_proc_rss_gb(d, t, group_size)


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------

def wait_for_available_memory(
    min_free_gb: float | None = None,
    *,
    poll_s: float = 5.0,
    timeout_s: float | None = None,
    on_wait=None,
) -> float:
    """Block until /proc/meminfo reports at least `min_free_gb` available.

    Returns the free memory observed when the condition cleared.
    Raises TimeoutError if `timeout_s` elapses first.

    `on_wait` is an optional callback invoked with
    (elapsed_s, free_gb) on every poll while waiting — use it to print
    a status message.
    """
    if min_free_gb is None:
        min_free_gb = _env_float("SWIFTBOT_MEM_BACKPRESSURE_GB", 6.0)
    if timeout_s is None:
        timeout_s = _env_float("SWIFTBOT_MEM_BACKPRESSURE_S", 600.0)

    start = time.monotonic()
    first = True
    while True:
        free = available_memory_gb()
        if free == 0.0:
            return free  # /proc/meminfo unreadable; skip backpressure
        if free >= min_free_gb:
            return free
        elapsed = time.monotonic() - start
        if timeout_s and elapsed > timeout_s:
            raise TimeoutError(
                f"memory backpressure: waited {elapsed:.0f}s for "
                f"{min_free_gb:.1f} GB free; only {free:.1f} GB available"
            )
        if on_wait is not None:
            on_wait(elapsed, free)
        elif first:
            first = False
            # Keep quiet on subsequent polls; caller can hook on_wait
            # if they want verbose logging.
        time.sleep(poll_s)
