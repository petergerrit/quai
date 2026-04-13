"""Pilot run: BI (binary icosahedral, |C|=120) Haar at t=50 ss=200 at d=2.

Sizes the deferred BI validation deferred in §4. ss=200 is chosen to be
roughly 10× our Clifford/hurwitz runs (20 samples) to get credible Q_T
statistics without committing to ss=10^4 before we know per-sample cost."""
import time
import numpy as np
from pathlib import Path
from swiftbot.stages import s3_efficiency as s3
from swiftbot.supervisor import ExtensionSpec
from swiftbot.kb.cache import Cache
from swiftbot.tools import groups as gmod

assert "BI" in gmod.REGISTRY, "BI must be registered in swiftbot.tools.groups"

spec = ExtensionSpec(
    kind="rnd", params={"pilot": "bi-t50-ss200"},
    rationale="BI Haar pilot to size ss=10^4 run",
)
with Cache(Path("sweep_runs/bi_pilot.db"), run_id="bi-pilot") as cache:
    t0 = time.time()
    recs = s3.evaluate_extension(
        spec, "BI", t=50, sample_size=200, cache=cache, timeout_s=14400,
    )
    dt = time.time() - t0

deltas = np.array([r.delta for r in recs])
qts = np.array([r.qt for r in recs if r.qt is not None])
print(f"BI t=50 ss=200: {dt:.1f}s  ({dt/200:.2f}s/sample)")
print(f"  δ: best={deltas.min():.4f}  mean={deltas.mean():.4f}  "
      f"std={deltas.std():.4f}  worst={deltas.max():.4f}")
print(f"  Q_T (best δ): {qts.min():.4f}  (Kesten-McKay floor 2.81)")
print(f"  est. ss=10^4 total time (linear extrapolation): {dt*50/3600:.1f}h")
