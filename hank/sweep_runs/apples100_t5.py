"""100-matrix apples-to-apples Haar panel on S216/S648/S1080 at t=5.

Addresses the \\itodo in §5 asking for ≥100 matrices to tighten the
apples-panel mean. Uses the same rng.default_rng(seed=42) as the 10-matrix
panel, so the first 10 matrices coincide with the 10-matrix panel."""
import time
import numpy as np
from pathlib import Path
from swiftbot.stages import s3_efficiency as s3
from swiftbot.supervisor import ExtensionSpec
from swiftbot.kb.cache import Cache

rng = np.random.default_rng(seed=42)


def haar_su(d: int = 3) -> np.ndarray:
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q / np.linalg.det(Q) ** (1 / d)


N_MATRICES = 100
matrices = [haar_su(3) for _ in range(N_MATRICES)]
stats: dict[str, list[float]] = {}

with Cache(Path("sweep_runs/apples100_t5.db"), run_id="apples100-t5") as cache:
    for grp in ("S216", "S648", "S1080"):
        stats[grp] = []
        t_group = time.time()
        for k, M in enumerate(matrices):
            spec = ExtensionSpec(
                kind="mat",
                params={"matrix": M.tolist(), "apple_id": k, "seed": 42},
                rationale=f"apples-100 apple{k:03d}",
            )
            t0 = time.time()
            recs = s3.evaluate_extension(
                spec, grp, t=5, sample_size=1, cache=cache, timeout_s=1800,
            )
            stats[grp].append(recs[0].delta)
            if k % 20 == 0 or k == N_MATRICES - 1:
                print(f"  {grp} {k+1:3d}/{N_MATRICES} delta={recs[0].delta:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        arr = np.array(stats[grp])
        print(f"== {grp} done ({time.time()-t_group:.0f}s)  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"best={arr.min():.4f}  worst={arr.max():.4f}", flush=True)

print("\n==== SUMMARY (apples-100 at t=5) ====")
for grp, arr in ((g, np.array(stats[g])) for g in ("S216", "S648", "S1080")):
    print(f"  {grp:6s} n={len(arr)}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"best={arr.min():.4f}  worst={arr.max():.4f}")
