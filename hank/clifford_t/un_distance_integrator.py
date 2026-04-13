import numpy as np
import sys
import os
import re
import math


# ---------------------------------------------------------------------------
# Random unitary generation — fully batched (no Python loop)
# ---------------------------------------------------------------------------

def random_un_batch(n: int, n_samples: int) -> np.ndarray:
    """
    Generate `n_samples` random U(N) matrices in one vectorised QR call.
    Returns Q : np.ndarray, shape (n_samples, n, n), dtype complex128
    """
    Z = (np.random.randn(n_samples, n, n)
         + 1j * np.random.randn(n_samples, n, n))
    Q, R = np.linalg.qr(Z)
    diag_R = R[:, np.arange(n), np.arange(n)]
    phases = diag_R / np.abs(diag_R)
    Q = Q * phases[:, None, :]
    return Q


def random_un(n: int) -> np.ndarray:
    return random_un_batch(n, 1)[0]


def random_sun(n: int) -> np.ndarray:
    Q = random_un(n)
    d = np.linalg.det(Q)
    return Q / (d ** (1.0 / n))


def random_u2() -> np.ndarray:
    return random_un_batch(2, 1)[0]


def random_su2() -> np.ndarray:
    v = np.random.randn(4)
    v /= np.linalg.norm(v)
    a = v[0] + 1j * v[1]
    b = v[2] + 1j * v[3]
    return np.array([[a, -np.conj(b)],
                     [b,  np.conj(a)]])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dist(A, B):
    return np.einsum('ij,kji->k', A, B).real


def get_ident_idx(B):
    return np.where(np.all(np.abs(B - np.eye(B.shape[1])) < 1e-10, axis=(1, 2)))[0]


def dist_checker(retrval, idx):
    return np.any(np.abs(np.delete(retrval, idx, axis=0)) > np.abs(retrval[idx]))


# ---------------------------------------------------------------------------
# Precomputed clifford view for fast matmul-based distance
# ---------------------------------------------------------------------------

def precompute_clifford_matrix(clifford: np.ndarray) -> np.ndarray:
    """
    Flatten and conjugate-transpose clifford matrices for use in matmul.

    The distance Re(Tr(C† S)) = Re( sum_{ij} conj(C[i,j]) * S[i,j] )
    = Re( clifford_flat @ sample_flat.T )

    where clifford_flat[l, :] = conj(C_l).ravel()  (shape: n_clifford x n²)
          sample_flat[k, :]   = S_k.ravel()          (shape: n_samples x n²)

    Using np.dot on these real-split representations avoids complex arithmetic
    overhead and lets BLAS handle the bulk of the work.
    """
    n_clifford, n, _ = clifford.shape
    # conj(C).ravel() per element -> (n_clifford, n*n) complex
    return clifford.conj().reshape(n_clifford, n * n)


def all_distances_matmul(samples: np.ndarray, clifford_flat: np.ndarray) -> np.ndarray:
    """
    Compute Re(Tr(C† S)) for all (sample, clifford) pairs via a single BLAS dgemm.

    samples       : (n_samples, n, n) complex
    clifford_flat : (n_clifford, n*n) complex  — from precompute_clifford_matrix

    Returns (n_samples, n_clifford) float64.

    Trick: split real/imag so the matmul runs on real arrays (BLAS dgemm is
    ~2x faster than zgemm for the same flop count on most hardware).
    Re(A · B†) = Re(A) @ Re(B†) - Im(A) @ Im(B†)
               = Re(A) @ Re(B)ᵀ + Im(A) @ Im(B)ᵀ   (because Re(B†)=Re(B)ᵀ,
                                                       Im(B†)=-Im(B)ᵀ but we
                                                       want Re of the product,
                                                       so signs work out)
    Here A = samples_flat, B = clifford_flat (already conjugated).
    Re(samples_flat @ clifford_flat†):
      clifford_flat already stores conj(C), so clifford_flat† = C (unconjugated).
      We want Re(samples_flat @ clifford_flat.conj().T)
            = Re_s @ Re_c.T + Im_s @ Im_c.T   (after distributing conjugate)
    
    Equivalently, since clifford_flat = conj(C_flat):
      Re(samples_flat @ C_flat.T) = Re_s @ Re_c.T - Im_s @ Im_c.T
    but Re(Tr(C†S)) = Re(clifford_flat @ sample_flat†) — let's just do it
    directly and correctly:
      all_dists[k,l] = Re( sum_m clifford_flat[l,m] * conj(sample_flat[k,m]) )
                     = Re_c @ Re_s.T + Im_c @ Im_s.T   (dot with conj of sample)
    Shape: (n_clifford, n_samples) -> transpose to (n_samples, n_clifford).
    """
    n_samples = samples.shape[0]
    n2 = clifford_flat.shape[1]

    s_flat = samples.reshape(n_samples, n2)   # (n_samples, n²)

    # Real-split matmul: (n_clifford, n²) x (n², n_samples) -> (n_clifford, n_samples)
    re_c = clifford_flat.real   # (n_clifford, n²)
    im_c = clifford_flat.imag
    re_s = s_flat.real          # (n_samples,  n²)
    im_s = s_flat.imag

    # Result[l,k] = Re(clifford_flat[l] · conj(s_flat[k]))
    #             = re_c[l] · re_s[k] + im_c[l] · im_s[k]
    result = re_c @ re_s.T + im_c @ im_s.T   # (n_clifford, n_samples)
    return result.T                           # (n_samples, n_clifford)


# ---------------------------------------------------------------------------
# Voronoi sampling — vectorised statistics (no per-element loop)
# ---------------------------------------------------------------------------

def _voronoi_raw_sums(all_dists: np.ndarray, n_clifford: int):
    """
    Return (counts, ws, ws2) — raw weighted sums for online aggregation —
    instead of computing means/stds per batch (saves redundant work).

    counts : (n_clifford,) int
    ws     : (n_clifford,) float  — sum of closest-distances per cell
    ws2    : (n_clifford,) float  — sum of squared closest-distances per cell
    """
    n_samples = all_dists.shape[0]
    closest = np.argmax(all_dists, axis=1)              # (n_samples,)
    vals    = all_dists[np.arange(n_samples), closest]  # (n_samples,)

    counts = np.bincount(closest, minlength=n_clifford).astype(int)
    ws     = np.bincount(closest, weights=vals,    minlength=n_clifford)
    ws2    = np.bincount(closest, weights=vals**2, minlength=n_clifford)
    return counts, ws, ws2


def _voronoi_stats(all_dists, n_clifford):
    """Legacy wrapper kept for API compatibility."""
    counts, ws, ws2 = _voronoi_raw_sums(all_dists, n_clifford)
    cb    = counts.astype(float)
    means = np.where(counts > 0, ws / cb, float('nan'))
    ex2   = np.where(counts > 0, ws2 / cb, float('nan'))
    stds  = np.where(counts > 0, np.sqrt(np.maximum(0.0, ex2 - means**2)), float('nan'))
    return counts, means, stds


def sample_un_voronoi(clifford, n_samples=1000):
    """Voronoi cell statistics for U(N). See sample_un_voronoi_fast for the
    accelerated path used in the main loop."""
    n = clifford.shape[-1]
    n_clifford = len(clifford)
    samples   = random_un_batch(n, n_samples)
    all_dists = np.einsum('kij,lji->kl', samples, clifford).real
    return _voronoi_stats(all_dists, n_clifford)


def sample_u2_voronoi(clifford, n_samples=1000):
    return sample_un_voronoi(clifford, n_samples)


def sample_un_dist_vectorized(clifford, n_samples=1000):
    n = clifford.shape[-1]
    idx = get_ident_idx(clifford)[0]
    samples   = random_un_batch(n, n_samples)
    all_dists = np.einsum('kij,lji->kl', samples, clifford).real
    identity_dists  = all_dists[:, idx]
    max_other_dists = np.max(np.delete(all_dists, idx, axis=1), axis=1)
    closest_mask    = identity_dists >= max_other_dists
    cnt    = np.sum(closest_mask)
    traces = np.trace(samples[closest_mask], axis1=1, axis2=2).real
    mean_trace = np.mean(traces) if len(traces) > 0 else float('nan')
    std_trace  = np.std(traces)  if len(traces) > 0 else float('nan')
    print(f"{cnt} of {n_samples} which corresponds to {cnt/n_samples:.4f}")
    print(f"Mean of Re(Tr(S)) for identity-closest: {mean_trace:.4f}")
    print(f"Std  of Re(Tr(S)) for identity-closest: {std_trace:.4f}")
    return cnt / n_samples, mean_trace, std_trace


def sample_u2_dist_vectorized(clifford, n_samples=1000):
    return sample_un_dist_vectorized(clifford, n_samples)


# ---------------------------------------------------------------------------
# Output file helpers
# ---------------------------------------------------------------------------

def parse_existing_output(path, n_clifford):
    """
    Read a previously written output file and recover the raw accumulators
    (counts, weighted_sum, weighted_sum2) so new samples can be added on top.

    We recover:
        weighted_sum  = counts * means
        weighted_sum2 = counts * (stds**2 + means**2)

    Returns (nsamp_prev, counts, weighted_sum, weighted_sum2) or None if the
    file cannot be parsed (e.g. does not exist or is malformed).
    """
    if not os.path.isfile(path):
        return None

    nsamp_prev    = None
    counts        = np.zeros(n_clifford, dtype=int)
    weighted_sum  = np.zeros(n_clifford)
    weighted_sum2 = np.zeros(n_clifford)

    # Regex patterns
    re_header  = re.compile(r'=== Total Results \((\d+) samples\) ===')
    re_row     = re.compile(
        r'^\s*(\d+)\s+(\d+)\s+[\d.]+\s+([\d.nan]+)\s+([\d.nan]+)\s*$')

    try:
        with open(path, 'r') as f:
            for line in f:
                m = re_header.search(line)
                if m:
                    nsamp_prev = int(m.group(1))
                    continue
                m = re_row.match(line)
                if m:
                    idx   = int(m.group(1))
                    cnt   = int(m.group(2))
                    mean  = float(m.group(3))
                    std   = float(m.group(4))
                    if idx < n_clifford:
                        counts[idx]        = cnt
                        weighted_sum[idx]  = cnt * mean
                        weighted_sum2[idx] = cnt * (std**2 + mean**2)
    except Exception as e:
        print(f"  [WARN] Could not parse existing output '{path}': {e}",
              file=sys.stderr)
        return None

    if nsamp_prev is None or np.sum(counts) == 0:
        return None

    print(f"  [RESUME] Found existing results: {nsamp_prev} samples across "
          f"{np.sum(counts > 0)} elements.", file=sys.stderr)
    return nsamp_prev, counts, weighted_sum, weighted_sum2


def write_output(path, counts_total, weighted_sum, weighted_sum2, nsamp_total, n_clifford):
    """Write the combined results to *path*, overwriting any previous content."""
    means = np.where(counts_total > 0, weighted_sum  / counts_total, float('nan'))
    ex2   = np.where(counts_total > 0, weighted_sum2 / counts_total, float('nan'))
    stds  = np.where(counts_total > 0,
                     np.sqrt(np.maximum(0.0, ex2 - means**2)),
                     float('nan'))

    lines = []
    lines.append(f"\n=== Total Results ({nsamp_total} samples) ===")
    lines.append(f"{'Element':>8} {'Count':>8} {'Ratio':>8} {'Mean Tr':>10} {'Std Tr':>10}")
    lines.append("-" * 50)
    for i in range(n_clifford):
        lines.append(
            f"{i:>8} {counts_total[i]:>8} {counts_total[i]/nsamp_total:>8.8f} "
            f"{means[i]:>10.8f} {stds[i]:>10.8f}")
    lines.append("-" * 50)
    lines.append(f"{'Total':>8} {np.sum(counts_total):>8} "
                 f"{np.sum(counts_total)/nsamp_total:>8.8f}")
    lines.append(f"\nMin count: {np.min(counts_total)} "
                 f"{np.min(counts_total)/nsamp_total:.8f} "
                 f"(element {np.argmin(counts_total)})")
    lines.append(f"Avg count: {np.mean(counts_total):.1f} "
                 f"{np.mean(counts_total)/nsamp_total:.8f} "
                 f"(closest element {np.argmin(np.abs(counts_total - np.mean(counts_total)))})")
    lines.append(f"Max count: {np.max(counts_total)} "
                 f"{np.max(counts_total)/nsamp_total:.8f} "
                 f"(element {np.argmax(counts_total)})")
    lines.append(f"Expected per cell (uniform): {nsamp_total/n_clifford:.1f} "
                 f"{1.0/n_clifford:.8f}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

file     = sys.argv[1]
clifford = np.load(file)
print(clifford.shape,        file=sys.stderr)
print(clifford[0],           file=sys.stderr)
print(len(clifford),         file=sys.stderr)

nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 10 * len(clifford)

# Output path is always derived from the input filename — no stdout redirect needed
stem     = os.path.splitext(os.path.basename(file))[0]
out_path = os.path.join("norm_data", f"data_{stem}")
os.makedirs("norm_data", exist_ok=True)

n_clifford = len(clifford)

# --- Try to resume from existing output ---
resume = parse_existing_output(out_path, n_clifford)
if resume is not None:
    nsamp_prev, counts_total, weighted_sum, weighted_sum2 = resume
else:
    nsamp_prev    = 0
    counts_total  = np.zeros(n_clifford, dtype=int)
    weighted_sum  = np.zeros(n_clifford)
    weighted_sum2 = np.zeros(n_clifford)

TARGET_GB       = 10
bytes_per_float = 8
n               = clifford.shape[-1]

MEM_BUDGET = (TARGET_GB * 1024**3) // (n**2 * bytes_per_float)
n_batches  = math.ceil((nsamp * n_clifford / MEM_BUDGET))
batch_size = nsamp // n_batches

print(f"Batches: {n_batches}, batch size: {batch_size}", file=sys.stderr)

# --- Precompute flattened clifford matrix once (used in every batch) ---
clifford_flat = precompute_clifford_matrix(clifford)   # (n_clifford, n²) complex

for batch in range(n_batches):
    print(f"Running batch {batch+1}/{n_batches}...", file=sys.stderr)

    samples   = random_un_batch(n, batch_size)
    all_dists = all_distances_matmul(samples, clifford_flat)
    counts_b, ws_b, ws2_b = _voronoi_raw_sums(all_dists, n_clifford)

    counts_total  += counts_b
    weighted_sum  += ws_b
    weighted_sum2 += ws2_b

nsamp_total = nsamp_prev + nsamp

# Write combined results to the output file
write_output(out_path, counts_total, weighted_sum, weighted_sum2,
             nsamp_total, n_clifford)
print(f"Done. Results written to: {out_path}  "
      f"({nsamp_total} total samples, {nsamp} new)", file=sys.stderr)
