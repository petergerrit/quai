"""BI Haar at d=2, t=50, sample_size=10^4 — paper-scale §4 run.

Extrapolated from the 200-sample pilot to ~20 h single-node wall time.
Writes to sweep_runs/bi_ss1e4.db; run_id 'bi-ss1e4-local'. Emits a
progress line every time qco's ss counter hits the reporting interval
(single subprocess; cannot stream per-sample progress from here)."""
import os
import sys
import time
from pathlib import Path

import numpy as np

# Avoid BLAS oversubscription for this single-subprocess run.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HANK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANK))

from swiftbot.stages import s3_efficiency as s3
from swiftbot.supervisor import ExtensionSpec
from swiftbot.kb.cache import Cache
from swiftbot.tools import groups as gmod

assert "BI" in gmod.REGISTRY, "BI must be registered; import swiftbot.tools.groups"

SAMPLE_SIZE = 10_000
T = 50

spec = ExtensionSpec(
    kind="rnd",
    params={"run": "bi-ss1e4-local"},
    rationale="BI paper-scale Haar ss=10^4 overnight, matches S{\\l}owik protocol",
)

db = HANK / "sweep_runs" / "bi_ss1e4.db"
print(f"START {time.strftime('%Y-%m-%d %H:%M:%S')}  BI t={T} ss={SAMPLE_SIZE}", flush=True)
print(f"DB: {db}", flush=True)

t_start = time.time()
SHARD_WORKERS = 4
print(f"shard_workers={SHARD_WORKERS} (per-shard ss≈{SAMPLE_SIZE // SHARD_WORKERS})", flush=True)

with Cache(db, run_id="bi-ss1e4-local") as cache:
    records = s3.evaluate_extension(
        spec, "BI",
        t=T, sample_size=SAMPLE_SIZE, cache=cache,
        timeout_s=None,  # no wall-time limit
        shard_workers=SHARD_WORKERS,
    )
elapsed = time.time() - t_start

deltas = np.array([r.delta for r in records])
qts = np.array([r.qt for r in records if r.qt is not None])
n = len(deltas)
print(f"END   {time.strftime('%Y-%m-%d %H:%M:%S')}  elapsed {elapsed/3600:.2f} h  ({elapsed/n:.2f} s/sample)", flush=True)
print(f"samples = {n}", flush=True)
print(f"δ     best={deltas.min():.6f}  mean={deltas.mean():.6f}  std={deltas.std():.6f}  worst={deltas.max():.6f}", flush=True)
print(f"Q_T   best={qts.min():.4f}  mean={qts.mean():.4f}  std={qts.std():.4f}", flush=True)
print(f"Q_opt for |C|=120 (BI) = 2.81  →  Q_T(best)/Q_opt = {qts.min()/2.81:.3f}", flush=True)
