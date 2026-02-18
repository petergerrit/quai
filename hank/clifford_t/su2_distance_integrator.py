import numpy as np
from numpy.linalg import det
import sys

def random_un(n: int) -> np.ndarray:
    """
    Generate a random U(N) matrix using QR decomposition (Haar measure).
    """
    Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(Z)
    # Adjust phases to ensure uniform (Haar) distribution
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * phases
    return Q


def random_sun(n: int) -> np.ndarray:
    """
    Generate a random SU(N) matrix (Haar measure).
    Take a random U(N) matrix and remove the overall phase
    by dividing by det^{1/N}.
    """
    Q = random_un(n)
    det = np.linalg.det(Q)
    # Remove global phase: multiply by det^{-1/N}
    Q = Q / (det ** (1.0 / n))
    return Q

def random_u2():
    """
    Generate a random U(2) matrix using QR decomposition.
    """
    # Generate random complex matrix
    Z = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    
    # QR decomposition gives a unitary matrix
    Q, R = np.linalg.qr(Z)
    
    # Adjust phases to ensure uniform distribution
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * phases
    
    return Q

def random_su2():
    """
    Generate a random SU(2) matrix.
    SU(2) matrices have the form:
    [[a, -b*], [b, a*]] where |a|^2 + |b|^2 = 1
    """
    v = np.random.randn(4)
    v = v / np.linalg.norm(v)
    
    a = v[0] + 1j * v[1]
    b = v[2] + 1j * v[3]
    
    return np.array([[a, -np.conj(b)],
                     [b,  np.conj(a)]])

def dist(A,B):
    return np.einsum('ij,kji->k', A, B).real

def get_ident_idx(B):
    return np.where(np.all(np.abs(B - np.eye(B.shape[1])) < 1e-10, axis=(1,2)))[0]

def dist_checker(retrval,idx):
    return np.any(np.abs(np.delete(retrval, idx, axis=0)) > np.abs(retrval[idx]))

def sample_un_dist_vectorized(clifford, n_samples=1000):
    n = clifford.shape[-1]  # infer matrix dimension from clifford array
    idx = get_ident_idx(clifford)
    # Generate all random U(N) matrices at once: shape (n_samples, n, n)
    samples = np.array([random_un(n) for _ in range(n_samples)])
    # Compute distances for all samples at once: shape (n_samples, n_clifford)
    all_dists = np.einsum('kij,lji->kl', samples, clifford).real
    # For each sample, check if identity IS the closest
    identity_dists = all_dists[:, idx[0]]          # shape (n_samples,)
    max_other_dists = np.max(
        np.delete(all_dists, idx[0], axis=1),       # shape (n_samples, n_clifford-1)
        axis=1
    )                                               # shape (n_samples,)
    # Boolean mask: True where identity IS the closest
    closest_mask = identity_dists >= max_other_dists
    cnt = np.sum(closest_mask)
    # Get the traces of the samples where identity IS closest
    traces_closest = np.trace(samples[closest_mask], axis1=1, axis2=2).real
    if len(traces_closest) > 0:
        mean_trace = np.mean(traces_closest)
        std_trace  = np.std(traces_closest)
    else:
        mean_trace = float('nan')
        std_trace  = float('nan')
    print(f"{cnt} of {n_samples} which corresponds to {cnt/n_samples:.4f}")
    print(f"Mean of Re(Tr(S)) for identity-closest: {mean_trace:.4f}")
    print(f"Std  of Re(Tr(S)) for identity-closest: {std_trace:.4f}")
    return cnt/n_samples, mean_trace, std_trace

def sample_u2_dist_vectorized(clifford, n_samples=1000):
    idx = get_ident_idx(clifford)
    
    # Generate all random U(2) matrices at once: shape (n_samples, 2, 2)
    samples = np.array([random_u2() for _ in range(n_samples)])
    
    # Compute distances for all samples at once: shape (n_samples, n_clifford)
    all_dists = np.einsum('kij,lji->kl', samples, clifford).real
    
    # For each sample, check if identity IS the closest
    identity_dists = all_dists[:, idx[0]]          # shape (n_samples,)
    max_other_dists = np.max(
        np.delete(all_dists, idx[0], axis=1),       # shape (n_samples, n_clifford-1)
        axis=1
    )                                               # shape (n_samples,)
    
    # Boolean mask: True where identity IS the closest
    # Identity is closest when no other element has a larger distance
    closest_mask = identity_dists >= max_other_dists  # FIXED: was > now >=
    cnt = np.sum(closest_mask)
    
    # Get the traces of the samples where identity IS closest
    traces_closest = np.trace(samples[closest_mask], axis1=1, axis2=2).real
    
    if len(traces_closest) > 0:
        mean_trace = np.mean(traces_closest)
        std_trace  = np.std(traces_closest)
    else:
        mean_trace = float('nan')
        std_trace  = float('nan')
    
    print(f"{cnt} of {n_samples} which corresponds to {cnt/n_samples:.4f}")
    print(f"Mean of Re(Tr(S)) for identity-closest: {mean_trace:.4f}")
    print(f"Std  of Re(Tr(S)) for identity-closest: {std_trace:.4f}")
    
    return cnt/n_samples, mean_trace, std_trace

def sample_u2_voronoi(clifford, n_samples=1000):
    """
    Compute Voronoi cell statistics for every element of the Clifford group.
    For each Clifford element, count how many random U(2) matrices are closest to it.
    """
    n_clifford = len(clifford)

    # Generate all random U(2) matrices at once: shape (n_samples, 2, 2)
    samples = np.array([random_u2() for _ in range(n_samples)])

    # Compute distances for all samples vs all clifford elements
    # shape (n_samples, n_clifford)
    all_dists = np.einsum('kij,lji->kl', samples, clifford).real

    # For each sample, find which Clifford element is closest (has max distance)
    closest_clifford = np.argmax(all_dists, axis=1)  # shape (n_samples,)

    # For each Clifford element, compute stats
#    print(f"{'Element':>8} {'Count':>8} {'Ratio':>8} {'Mean Tr':>10} {'Std Tr':>10}")
#    print("-" * 50)

    counts = np.zeros(n_clifford, dtype=int)
    means  = np.zeros(n_clifford)
    stds   = np.zeros(n_clifford)

    for i in range(n_clifford):
        # Mask for samples closest to clifford element i
        mask = closest_clifford == i
        counts[i] = np.sum(mask)

        if counts[i] > 0:
            # Compute Re(Tr(C_i^dag * S)) for samples in this Voronoi cell
            # i.e. the distances to clifford element i for those samples
            cell_samples = samples[mask]  # shape (counts[i], 2, 2)

            # Compute Re(Tr(C_i * S)) for each sample in this cell
            cell_dists = np.einsum('ij,kji->k', clifford[i], cell_samples).real

            means[i] = np.mean(cell_dists)
            stds[i]  = np.std(cell_dists)
        else:
            means[i] = float('nan')
            stds[i]  = float('nan')

#        print(f"{i:>8} {counts[i]:>8} {counts[i]/n_samples:>8.5f} {means[i]:>10.5f} {stds[i]:>10.5f}")

#    print("-" * 50)
#    print(f"{'Total':>8} {np.sum(counts):>8} {np.sum(counts)/n_samples:>8.4f}")
#    print(f"\nMin count: {np.min(counts)} (element {np.argmin(counts)})")
#    print(f"Max count: {np.max(counts)} (element {np.argmax(counts)})")
#    print(f"Expected per cell (uniform): {n_samples/n_clifford:.1f}")

    return counts, means, stds

def sample_un_voronoi(clifford, n_samples=1000):
    """
    Compute Voronoi cell statistics for every element of the Clifford group.
    For each Clifford element, count how many random U(N) matrices are closest to it.
    """
    n = clifford.shape[-1]  # infer matrix dimension from clifford array
    n_clifford = len(clifford)
    # Generate all random U(N) matrices at once: shape (n_samples, n, n)
    samples = np.array([random_un(n) for _ in range(n_samples)])
    # Compute distances for all samples vs all clifford elements
    # shape (n_samples, n_clifford)
    all_dists = np.einsum('kij,lji->kl', samples, clifford).real
    # For each sample, find which Clifford element is closest (has max distance)
    closest_clifford = np.argmax(all_dists, axis=1)  # shape (n_samples,)
    # For each Clifford element, compute stats
    counts = np.zeros(n_clifford, dtype=int)
    means  = np.zeros(n_clifford)
    stds   = np.zeros(n_clifford)
    for i in range(n_clifford):
        mask = closest_clifford == i
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            cell_samples = samples[mask]  # shape (counts[i], n, n)
            cell_dists = np.einsum('ij,kji->k', clifford[i], cell_samples).real
            means[i] = np.mean(cell_dists)
            stds[i]  = np.std(cell_dists)
        else:
            means[i] = float('nan')
            stds[i]  = float('nan')
    return counts, means, stds

file = sys.argv[1]
nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
clifford = np.load(file)
print(clifford.shape)  # (24, 2, 2)
print(clifford[0])     # First group element

n_batches = 100
batch_size = nsamp // n_batches

counts_total = np.zeros(len(clifford), dtype=int)
means_total  = np.zeros(len(clifford))
stds_total   = np.zeros(len(clifford))

# Store batch results
batch_counts = []
batch_means  = []

for batch in range(n_batches):
    print(f"Running batch {batch+1}/{n_batches}...")
    counts_b, means_b, stds_b = sample_un_voronoi(clifford, n_samples=batch_size)
    
    # Replace nan with 0 before accumulating
    means_b = np.nan_to_num(means_b, nan=0.0)
    
    # Store for std computation later
    batch_counts.append(counts_b)
    batch_means.append(means_b)
    
    counts_total += counts_b
    means_total  += means_b * counts_b  # Weighted sum

# Normalize means by total counts
means_total = np.where(counts_total > 0, means_total / counts_total, float('nan'))

# Recompute std from stored batch results
for counts_b, means_b in zip(batch_counts, batch_means):
    stds_total += np.where(
        counts_b > 0,
        counts_b * ((means_b - means_total)**2),
        0.0
    )

# Normalize stds
stds_total = np.where(counts_total > 0, np.sqrt(stds_total / counts_total), float('nan'))

counts, means, stds = counts_total, means_total, stds_total

print(f"\n=== Total Results ({nsamp} samples) ===")
print(f"{'Element':>8} {'Count':>8} {'Ratio':>8} {'Mean Tr':>10} {'Std Tr':>10}")
print("-" * 50)
for i in range(len(clifford)):
    print(f"{i:>8} {counts[i]:>8} {counts[i]/nsamp:>8.6f} {means[i]:>10.6f} {stds[i]:>10.6f}")
print("-" * 50)
print(f"{'Total':>8} {np.sum(counts):>8} {np.sum(counts)/nsamp:>8.6f}")
print(f"\nMin count: {np.min(counts)} {np.min(counts)/nsamp:.6f} (element {np.argmin(counts)})")
print(f"Avg count: {np.mean(counts):.1f} {np.mean(counts)/nsamp:.6f} (closest element {np.argmin(np.abs(counts - np.mean(counts)))})")
print(f"Max count: {np.max(counts)} {np.max(counts)/nsamp:.6f} (element {np.argmax(counts)})")
print(f"Expected per cell (uniform): {nsamp/len(clifford):.1f} {1.0/len(clifford):.6f}")

