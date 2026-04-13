"""Calibration probe: single S216 d=3 t=10 ss=1 eval to anchor the
t=5 -> t=10 scale factor for Tier 2 planning."""
import time
from pathlib import Path
from swiftbot.stages import s3_efficiency as s3
from swiftbot.supervisor import ExtensionSpec
from swiftbot.kb.cache import Cache

spec = ExtensionSpec(
    kind="howard_vala", params={"z": 2, "gamma": 1, "eps": 0},
    rationale="calibration probe",
)
with Cache(Path("sweep_runs/calib_t10.db"), run_id="calib-t10") as cache:
    t0 = time.time()
    recs = s3.evaluate_extension(
        spec, "S216", t=10, sample_size=1, cache=cache, timeout_s=14400,
    )
    dt = time.time() - t0
    print(f"RESULT: S216 HV(2,1,0) t=10 ss=1 took {dt:.1f}s  delta={recs[0].delta:.6f}  qt={recs[0].qt:.4f}")
