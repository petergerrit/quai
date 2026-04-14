"""Tests for swiftbot.stages.mem_safety — the three-layer memory defense
(pre-flight budget check, per-subprocess RLIMIT_AS, runtime backpressure)
introduced after the Tier-2 OOM lockup on lenore (2026-04-13)."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from swiftbot.stages import mem_safety


# ---------------------------------------------------------------------------
# Per-proc RSS model
# ---------------------------------------------------------------------------

def test_per_proc_rss_scaling_d3() -> None:
    """At d=3, per-proc RSS scales ~linearly in |C| and ~quadratically in t.
    Anchor: t=10, |C|=216 ≈ 4 GB (Tier-2 observation)."""
    r_216 = mem_safety.estimate_per_proc_rss_gb(d=3, t=10, group_size=216)
    r_1080 = mem_safety.estimate_per_proc_rss_gb(d=3, t=10, group_size=1080)
    # t=10 on S216: 2 + 0.01 * 216 = 4.16 GB, matches calibration anchor
    assert 3.5 <= r_216 <= 5.0, f"S216 @ t=10 expected ~4 GB, got {r_216}"
    # t=10 on S1080 should be materially larger (we crashed at that size)
    assert r_1080 > 10.0, f"S1080 @ t=10 expected >10 GB, got {r_1080}"
    assert r_1080 > 2.5 * r_216, "per-proc RSS must scale with |C|"


def test_per_proc_rss_scaling_t() -> None:
    """Quadratic-ish growth in t for d=3."""
    r_t5 = mem_safety.estimate_per_proc_rss_gb(d=3, t=5, group_size=1080)
    r_t10 = mem_safety.estimate_per_proc_rss_gb(d=3, t=10, group_size=1080)
    r_t15 = mem_safety.estimate_per_proc_rss_gb(d=3, t=15, group_size=1080)
    assert r_t10 > r_t5
    assert r_t15 > r_t10
    # At least quadratic growth in t for the dominant (|C|-scaled) term.
    assert (r_t15 - 2.0) > 2.0 * (r_t10 - 2.0), "per-proc RSS should grow at least ∝ t²"


def test_per_proc_rss_env_override() -> None:
    """SWIFTBOT_MEM_PER_PROC_GB pins the estimate regardless of d/t/|C|."""
    with patch.dict(os.environ, {"SWIFTBOT_MEM_PER_PROC_GB": "7.5"}):
        for d in (2, 3, 4):
            r = mem_safety.estimate_per_proc_rss_gb(d=d, t=10, group_size=1080)
            assert r == pytest.approx(7.5)


def test_per_proc_rss_d2_cheaper_than_d3() -> None:
    """d=2 should be substantially cheaper than d=3 for same |C|,t."""
    r2 = mem_safety.estimate_per_proc_rss_gb(d=2, t=10, group_size=120)
    r3 = mem_safety.estimate_per_proc_rss_gb(d=3, t=10, group_size=120)
    assert r2 < r3, "d=2 must be cheaper than d=3 per proc"


# ---------------------------------------------------------------------------
# Panel peak estimator
# ---------------------------------------------------------------------------

def test_panel_peak_pins_to_largest_group() -> None:
    """The estimator assumes worst case: every worker on the largest group."""
    budget = mem_safety.estimate_panel_peak_gb(
        group_sizes=[216, 648, 1080],
        extension_kinds=["rnd", "angle"],
        d=3, t=10, workers=8, shard_workers=4, rnd_samples=100,
    )
    # per_proc pinned to largest group (|C|=1080)
    r_1080 = mem_safety.estimate_per_proc_rss_gb(d=3, t=10, group_size=1080)
    assert budget.peak_per_proc_gb == pytest.approx(r_1080)
    # peak_procs is workers * shard_workers because panel has a rnd entry
    assert budget.peak_procs == 8 * 4


def test_panel_peak_without_rnd_collapses_to_workers() -> None:
    """No rnd → shard_workers doesn't apply, peak_procs == workers."""
    budget = mem_safety.estimate_panel_peak_gb(
        group_sizes=[216, 1080],
        extension_kinds=["angle", "howard_vala"],
        d=3, t=10, workers=8, shard_workers=4, rnd_samples=100,
    )
    assert budget.peak_procs == 8  # shard_workers ignored


def test_panel_peak_tier2_would_have_caught_the_lockup() -> None:
    """The exact Tier-2 configuration that OOM-locked lenore on 2026-04-13
    must fail the budget check on a 125 GB machine."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SWIFTBOT_MEM_OVERHEAD_GB", None)
        os.environ.pop("SWIFTBOT_MEM_SAFETY_FRAC", None)
        # Simulate lenore: 125 GB total, ~120 GB available at panel start.
        with patch.object(mem_safety, "available_memory_gb", return_value=120.0):
            with patch.object(mem_safety, "total_memory_gb", return_value=125.0):
                budget = mem_safety.estimate_panel_peak_gb(
                    group_sizes=[216, 648, 1080],
                    extension_kinds=(
                        ["howard_vala"] * 3
                        + ["angle"] * 3
                        + ["rnd"] * 2
                    ),
                    d=3, t=10,
                    workers=8, shard_workers=4,
                    rnd_samples=100,
                )
    # Peak ≈ 32 × ~13 GB + 4 GB overhead ≈ 420 GB; budget ≈ 84 GB (70% of 120).
    # The check must flag this as unsafe.
    assert budget.total_gb > budget.available_gb * budget.safety_frac
    assert budget.safe is False


def test_check_budget_raises_when_unsafe() -> None:
    """check_budget_or_raise must raise MemoryBudgetExceeded on unsafe
    panels and include the reason string."""
    with patch.object(mem_safety, "available_memory_gb", return_value=8.0):
        with patch.object(mem_safety, "total_memory_gb", return_value=8.0):
            with pytest.raises(mem_safety.MemoryBudgetExceeded) as exc_info:
                mem_safety.check_budget_or_raise(
                    group_sizes=[1080],
                    extension_kinds=["rnd"],
                    d=3, t=10, workers=8, shard_workers=4,
                    rnd_samples=100,
                )
    assert "peak" in str(exc_info.value)
    assert "1080" in str(exc_info.value) or "|C|" in str(exc_info.value)
    assert "override" in str(exc_info.value).lower() or "ignore" in str(exc_info.value).lower()


def test_check_budget_override_flag_bypasses_raise() -> None:
    """override=True lets the run proceed even when unsafe."""
    with patch.object(mem_safety, "available_memory_gb", return_value=8.0):
        # No exception expected
        report = mem_safety.check_budget_or_raise(
            group_sizes=[1080],
            extension_kinds=["rnd"],
            d=3, t=10, workers=8, shard_workers=4,
            rnd_samples=100, override=True,
        )
    assert report.safe is False
    assert "peak" in report.reason


def test_check_budget_passes_on_modest_panel() -> None:
    """A small panel on a well-provisioned machine must pass."""
    with patch.object(mem_safety, "available_memory_gb", return_value=120.0):
        report = mem_safety.check_budget_or_raise(
            group_sizes=[216],
            extension_kinds=["angle"],
            d=3, t=5, workers=2, shard_workers=1,
        )
    assert report.safe is True


# ---------------------------------------------------------------------------
# Per-subprocess cap
# ---------------------------------------------------------------------------

def test_per_subprocess_cap_has_margin_over_estimate() -> None:
    """RLIMIT cap must exceed the model estimate so benign overshoots don't
    fail, but bound by a multiplier so catastrophic ones do."""
    est = mem_safety.estimate_per_proc_rss_gb(d=3, t=10, group_size=1080)
    cap = mem_safety.per_subprocess_cap_gb(d=3, t=10, group_size=1080)
    assert cap > est
    assert cap < 3.0 * est  # default multiplier is 1.5, must stay reasonable


def test_rlimit_preexec_returns_callable() -> None:
    """rlimit_preexec returns a callable suitable for subprocess preexec_fn.
    We don't actually fork here, just verify it can be called."""
    fn = mem_safety.rlimit_preexec(10.0)
    assert callable(fn)
    # Invoking the preexec fn in the test process is a no-op-or-raises-
    # safely guarded — shouldn't crash the test.
    try:
        fn()
    except (ValueError, OSError):
        pass  # acceptable: process may not be allowed to lower its limit


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------

def test_wait_returns_immediately_when_enough_memory() -> None:
    """If available_memory_gb reports plenty of memory, return without sleep."""
    with patch.object(mem_safety, "available_memory_gb", return_value=100.0):
        free = mem_safety.wait_for_available_memory(
            min_free_gb=6.0, poll_s=0.1, timeout_s=5.0,
        )
    assert free == pytest.approx(100.0)


def test_wait_times_out_on_persistent_low_memory() -> None:
    """If memory never recovers, wait_for_available_memory must time out."""
    with patch.object(mem_safety, "available_memory_gb", return_value=1.0):
        with pytest.raises(TimeoutError):
            mem_safety.wait_for_available_memory(
                min_free_gb=6.0, poll_s=0.05, timeout_s=0.15,
            )


def test_wait_short_circuits_if_meminfo_unreadable() -> None:
    """If /proc/meminfo returns 0.0 (non-Linux / sandboxed), return
    immediately without blocking the run."""
    with patch.object(mem_safety, "available_memory_gb", return_value=0.0):
        free = mem_safety.wait_for_available_memory(
            min_free_gb=6.0, poll_s=0.1, timeout_s=5.0,
        )
    assert free == 0.0


# ---------------------------------------------------------------------------
# Integration: /proc/meminfo real probe (smoke test; may skip off-Linux)
# ---------------------------------------------------------------------------

def test_real_meminfo_probe_returns_something_positive() -> None:
    """On a Linux CI, /proc/meminfo should yield a positive number."""
    v = mem_safety.available_memory_gb()
    total = mem_safety.total_memory_gb()
    if total == 0.0:
        pytest.skip("/proc/meminfo unreadable; non-Linux host")
    assert v > 0
    assert total >= v  # available is bounded by total
