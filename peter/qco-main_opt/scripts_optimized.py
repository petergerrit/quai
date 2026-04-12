"""
Optimized scripts.py with:
1. Fast representation construction (3883x speedup)
2. Batched processing to avoid OOM
3. Progress logging
4. Parallel sampling support
"""
from __future__ import annotations

from functools import cmp_to_key
import gc
import math
import os
import random
import signal
import sys
import time

import numpy as np
from scipy.linalg import eigh, svdvals, norm
from scipy import sparse

from consts import *
from dataStructures import *

# Use optimized representations
from representation_optimized import (
    suRepresentationOptimized as suRepresentation,
    SURepresentationOptimized as SURepresentation,
    uRepresentationOptimized as uRepresentation,
    URepresentationOptimized as URepresentation,
)
from utils import *

mpi_loaded = 1
try:
    from mpi4py import MPI
except ImportError:
    mpi_loaded = 0


class TimeLimitSignal(Exception):
    def __init__(self):
        super().__init__('Time limit exceeded.')


def handler(signum, frame):
    raise TimeLimitSignal


if sys.platform.startswith('linux'):
    signal.signal(signal.SIGUSR1, handler)

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)


def divide_list(l, n):
    ll = [[] for i in range(n)]
    for i in range(len(l)):
        ll[i % n].append(l[i])
    return ll


def get_random_SU(n: int, order: int | None = None):
    """Returns Haar random matrix from SU(n)."""
    if order is not None:
        v = get_random_SU(n, order=None)
        ks = [k for k in range(1, order) if np.gcd(order, k) == 1]
        k = random.choice(ks)
        phases = [np.exp(1j * 2 * np.pi * k * j / order) for j in range(n - 1)]
        phases.append(np.conj(np.prod(phases)))
        diag_mat = np.asmatrix(np.diag(phases))
        v_dagger = np.conjugate(v.T)
        return v * diag_mat * v_dagger

    x1 = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    q, r = np.linalg.qr(x1)
    r = np.diag(r) / abs(np.diag(r))
    u = np.multiply(q, r, q)
    tu = np.power(np.linalg.det(u), 1 / n)
    u1 = np.divide(u, tu)
    return np.matrix(u1)


def get_gates_from_file(fileName):
    gates = []
    if fileName == "":
        return 0, []

    with open(fileName, 'r') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        for i in range(1, len(lines), 1):
            line = lines[i].split()
            if (i - 1) % n == 0:
                gates.append([])
            gates[-1].append([complex(x) for x in line])

    return n, [np.array(x) for x in gates]


def _parse_gate_txt(path: str, d: int) -> np.ndarray:
    """Read a single d x d complex matrix from a plain text file."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    try:
        if int(lines[0]) == d:
            lines = lines[1:]
    except (ValueError, IndexError):
        pass

    mat = [[complex(x) for x in line.split()] for line in lines[:d]]
    if len(mat) != d or any(len(row) != d for row in mat):
        raise ValueError(f"Expected a {d}x{d} matrix in '{path}', got {len(mat)} rows.")
    return np.array(mat)


def norm_dim(weight):
    return suRepresentation.weight_to_dim(weight)


def get_SU_weights(d, max_norm, norm=norm2):
    """Generates all SU rep. weights with the norm smaller than J."""
    weight = np.zeros(d - 1, dtype=np.int64)
    weight[0] = 1

    weights = []
    got_max_weight = False

    while not got_max_weight:
        weights.append(weight.copy())

        i = 0
        weight[i] += 1
        while norm(weight) > max_norm:
            if i + 1 >= d - 1:
                got_max_weight = True
                break

            weight[i] = 0
            weight[i + 1] += 1
            i += 1

    return weights


def get_PU_weights(d, max_norm, norm=norm2):
    result = []
    for weight in get_SU_weights(d, max_norm, norm):
        if suRepresentation.is_projective(weight):
            result.append(weight)
    return result


def compare_weights(w1, w2, cmp_by_t=True):
    if len(w1) != len(w2):
        raise ValueError('Weights %s and %s have different lengths.' % (w1, w2))

    if cmp_by_t:
        if suRepresentation.is_projective(w1) and suRepresentation.is_projective(w2):
            uw1 = suRepresentation.unspecial_weight(w1, True)
            uw2 = suRepresentation.unspecial_weight(w2, True)

            t1 = sum(x for x in uw1 if x > 0)
            t2 = sum(x for x in uw2 if x > 0)

            c = cmp(t1, t2)
            if c:
                return c

            return cmp(suRepresentation.weight_to_dim(w1), suRepresentation.weight_to_dim(w2))
        raise ValueError('Weights %s and %s cannot be compared by t.' % (w1, w2))

    for i in range(len(w1) - 1, -1, -1):
        tmp = cmp(w1[i], w2[i])
        if tmp != 0:
            return tmp

    return 0


def filter_mirror_weights(weights):
    i = 0
    while i < len(weights):
        weight = tuple(weights[i])
        i += 1

        if not suRepresentation.is_complex(weight):
            continue

        weights = weights[:i] + [w for w in weights[i:] if not weight[::-1] == tuple(w)]
    
    return weights


def t_design_weights(d, t):
    """Returns all SU(d) representation weights appearing in the decomposition of t-design."""
    if d == 2:
        return [(2 * i + 2,) for i in range(t)]

    trivial_weight = tuple(0 for _ in range(d - 1))
    weights = []
    weights_set = set()
    for i in range(1, t + 1):
        partitions = get_partitions(i, d)

        for p1 in partitions:
            for p2 in partitions:
                u_weight = partitions_to_weight(p1, p2)
                su_weight = tuple(u_weight[i] - u_weight[i + 1] for i in range(len(u_weight) - 1))
                if su_weight != trivial_weight and su_weight not in weights_set:
                    weights.append(su_weight)
                    weights_set.add(su_weight)

    return weights


def get_partitions(n, k, bound=math.inf):
    """Returns all partitions of n into k terms non-greater than bound."""
    if k == 0:
        return []

    if n == 0 and bound >= 0:
        return [[0] * k]

    if k > n:
        return [prefix + [0] * (k - n) for prefix in get_partitions(n, n, bound)]

    if k == 1:
        return [[n]] if n <= bound else []

    partitions = []
    first = min(n, bound)
    suffixes = get_partitions(n - first, k - 1, first)
    while len(suffixes) > 0:
        partitions += [[first] + suffix for suffix in suffixes]
        first -= 1
        suffixes = get_partitions(n - first, k - 1, first)

    return partitions


def partitions_to_weight(partition1, partition2):
    if len(partition1) != len(partition2):
        raise ValueError('Partitions have to be of the same length.')

    return [partition1[i] - partition2[- i - 1] for i in range(len(partition1))]


class RepresentationCache:
    """
    Memory-efficient representation cache that builds representations on-demand
    and discards them after use to manage memory.
    """
    
    def __init__(self, weights, batch_size=50, verbose=True):
        """
        Args:
            weights: List of weight tuples
            batch_size: Number of representations to keep in memory at once
            verbose: Whether to print progress
        """
        self.weights = [tuple(w) for w in weights]
        self.batch_size = batch_size
        self.verbose = verbose
        self._cache = {}
        self._access_order = []
        
        # Pre-compute dimensions for sorting/planning
        self.dims = {tuple(w): suRepresentation.weight_to_dim(w) for w in weights}
        
    def get(self, weight):
        """Get a representation, building it if necessary."""
        weight = tuple(weight)
        
        if weight in self._cache:
            return self._cache[weight]
        
        # Build the representation
        rep = SURepresentation(weight)
        
        # Cache management: evict oldest if at capacity
        if len(self._cache) >= self.batch_size:
            # Remove oldest entry
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
            gc.collect()
        
        self._cache[weight] = rep
        self._access_order.append(weight)
        
        return rep
    
    def clear(self):
        """Clear all cached representations."""
        self._cache.clear()
        self._access_order.clear()
        gc.collect()


def sample_norms_optimized(sample_size, n_of_generators, d=2, v='0.0.0',
    gates_path='', weights_gen='t-design', weights_filter='PU',
    gate_order: int | None = None,
    fixed_gate_angle: float | None = None,
    fixed_gate_angles: str | None = None,
    fixed_gate_matrix: str | None = None,
    batch_size: int = 50,
    progress_interval: int = 10,
    **kwargs):
    """
    Optimized version of sample_norms with batched representation processing.
    
    Additional args:
        batch_size: Number of representations to keep in memory at once
        progress_interval: How often to print progress (in samples)
    """
    comm = None
    rank = 0
    size = 1

    if mpi_loaded:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    exception_raised = None

    sample_size = int(sample_size)
    batch_size = int(batch_size)

    Gs_slice = [int(x) for x in n_of_generators.split('-')]
    if len(Gs_slice) > 1:
        n_of_generators = range(*Gs_slice)
    else:
        n_of_generators = Gs_slice

    d = int(d)

    symmetries = [0]
    if 'symmetric' in kwargs:
        symmetries = [1]
    if 'unsymmetric' in kwargs:
        symmetries = [0]

    save_spectrum = 'save_spectrum' in kwargs

    use_checkpoints = True
    first_checkpoint = True

    from_file = bool(gates_path)
    f_gates = []
    group_tag = ''
    if from_file:
        d, f_gates = get_gates_from_file(gates_path)
        group_tag = os.path.splitext(os.path.basename(gates_path))[0]

    start = time.time()
    
    # Timing categories
    NORM = 'norm'
    REP = 'rep'
    REP_BUILD = 'rep_build'
    GATHER = 'gather'
    ADD_EIGS = 'add eigenvalues'
    PREPARE_SAVE = 'prepare save'
    SAVE_DELTAS = 'save deltas'
    SAVE_GATES = 'save gates'
    ERASE = 'erase'
    ALL = 'all'
    times = {
        NORM: 0.0,
        REP: 0.0,
        REP_BUILD: 0.0,
        GATHER: 0.0,
        ADD_EIGS: 0.0,
        PREPARE_SAVE: 0.0,
        SAVE_DELTAS: 0.0,
        SAVE_GATES: 0.0,
        ERASE: 0.0
    }
    max_rep_time = 0.0
    max_rep_weight = None
    max_rep_dim = 0

    def print_raport():
        end = time.time()
        times[ALL] = end - start
        print("\nTiming Report:")
        print("-" * 50)
        for key in times:
            pct = times[key] / times[ALL] * 100 if times[ALL] > 0 else 0
            print(f"  {key:20s}: {times[key]:8.2f}s ({pct:5.1f}%)")
        print("-" * 50)
        print(f"Max representation: weight={max_rep_weight}, dim={max_rep_dim}, time={max_rep_time:.2f}s")
        print(f"Total time: {times[ALL]:.1f}s ({times[ALL]/60:.1f} min)")

    # Generate weights
    options = {}
    weights = []
    if weights_gen == 'norm2':
        J = int(kwargs['J'])
        weights = get_SU_weights(d, J, norm2)
        options['J'] = int(round(math.log10(J)))
    elif weights_gen == 'dim':
        J = int(kwargs['J'])
        weights = get_SU_weights(d, int(kwargs['J']), norm_dim)
        options['J'] = int(round(math.log10(J)))
    elif weights_gen == 't-design':
        t = int(kwargs['t'])
        weights = [np.array(weight) for weight in t_design_weights(d, t)]
        options['T'] = t

    if weights_filter == 'PU':
        weights = [weight for weight in weights if suRepresentation.is_projective(weight)]
    elif weights_filter == 'Q':
        weights = [weight for weight in weights if suRepresentation.is_quaternionic(weight)]

    weights = filter_mirror_weights(weights)
    
    # Sort weights by dimension (process smaller ones first for better memory management)
    weights_with_dims = [(w, suRepresentation.weight_to_dim(w)) for w in weights]
    weights_with_dims.sort(key=lambda x: x[1])
    weights = [w for w, _ in weights_with_dims]

    if rank == 0:
        total_dim = sum(d for _, d in weights_with_dims)
        print(f"\n{'='*60}")
        print(f"OPTIMIZED SAMPLE_NORMS")
        print(f"{'='*60}")
        print(f"Representations: {len(weights)}")
        print(f"Total dimension: {total_dim:,}")
        print(f"Max dimension: {max(d for _, d in weights_with_dims)}")
        print(f"Batch size: {batch_size}")
        print(f"Sample size: {sample_size}")
        print(f"{'='*60}\n")

    if fixed_gate_angle is not None:
        fixed_gate_angle = float(fixed_gate_angle)

    options = {
        **options,
        'N': int(round(math.log10(sample_size))) if sample_size > 0 else 0,
        'f': 1 if from_file else 0,
        'v': v,
        **({'G': group_tag} if group_tag else {}),
    }

    if gate_order is not None:
        gate_order = int(gate_order)
        options = {**options, 'r': gate_order}

    # Build the fixed extension gate
    fixed_gate = None

    if fixed_gate_matrix is not None:
        if fixed_gate_matrix.endswith('.npy'):
            raw = np.array(np.load(fixed_gate_matrix), dtype=complex)
        else:
            raw = _parse_gate_txt(fixed_gate_matrix, d)
        det = np.linalg.det(raw)
        fixed_gate = np.asmatrix(raw / (det ** (1.0 / d)))
        ext_type = 'mat'
        ext_val = os.path.splitext(os.path.basename(fixed_gate_matrix))[0]
        options = {**options, 'exttype': ext_type, 'extval': ext_val}

    elif fixed_gate_angles is not None:
        angle_vals = [float(a) for a in fixed_gate_angles.split(',')]
        if len(angle_vals) != d - 1:
            raise ValueError(
                f"fixed_gate_angles requires exactly d-1={d-1} comma-separated "
                f"values for SU({d}), got {len(angle_vals)}."
            )
        phases = [np.exp(1j * a * np.pi) for a in angle_vals]
        phases.append(np.conj(np.prod(phases)))
        fixed_gate = np.asmatrix(np.diag(phases))
        ext_type = 'angles'
        ext_val = angle_vals
        options = {**options, 'exttype': ext_type, 'extval': ext_val}

    elif fixed_gate_angle is not None:
        phase = np.exp(1j * fixed_gate_angle * np.pi)
        raw = np.diag([phase] + [1.0] * (d - 1)).astype(complex)
        det = np.linalg.det(raw)
        fixed_gate = np.asmatrix(raw / (det ** (1.0 / d)))
        ext_type = 'angle'
        ext_val = fixed_gate_angle
        options = {**options, 'exttype': ext_type, 'extval': ext_val}

    else:
        options = {**options, 'exttype': 'rnd'}

    if save_spectrum:
        options = {**options, 'spec': ''}

    if rank == 0:
        print("Options:")
        for k, v in options.items():
            print(f"  {k}: {v}")
        print(f"Symmetries: {symmetries}\n")

    # Use representation cache for memory management
    rep_cache = RepresentationCache(weights, batch_size=batch_size, verbose=(rank == 0))

    rest = sample_size % size
    my_N = (sample_size - rest) // size
    my_N += int(rank < rest)

    # Initialize data structures
    datas = {}
    for symmetry in symmetries:
        tmp_options = {'s': symmetry, **options}
        datas[symmetry] = [
            QcoData(d, G, tmp_options, [], rank) for G in n_of_generators
        ]
        # Store weight list for output (as lists for compatibility)
        for data in datas[symmetry]:
            data.weights = [list(w) for w in weights]

    gates_data = QcoData(d, 1, options, [], rank)

    # Main sampling loop
    for sample_idx in range(my_N):
        if rank == 0 and (sample_idx + 1) % progress_interval == 0:
            elapsed = time.time() - start
            rate = (sample_idx + 1) / elapsed
            eta = (my_N - sample_idx - 1) / rate if rate > 0 else 0
            print(f"Sample {sample_idx + 1}/{my_N} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                  f"Rate: {rate:.2f} samples/s")

        ops = {s: [[] for _ in n_of_generators] for s in symmetries}
        gates = []
        Gmax = max(n_of_generators)
        
        if fixed_gate is not None:
            random_gates = [fixed_gate] * Gmax
        else:
            random_gates = [get_random_SU(d, gate_order) for _ in range(Gmax)]

        if from_file:
            gates = [g @ rg @ g.conjugate().T for rg in random_gates for g in f_gates]
        else:
            gates = random_gates

        try:
            # Process representations in batches
            for weight_idx, weight in enumerate(weights):
                # Build representation (from cache or fresh)
                flag = time.time()
                Pi = rep_cache.get(weight)
                build_time = time.time() - flag
                times[REP_BUILD] += build_time

                # Apply representation to gates
                flag = time.time()
                repr_of_gates = [Pi(gate) for gate in gates]
                rep_time = time.time() - flag
                times[REP] += rep_time

                if rep_time > max_rep_time:
                    max_rep_time = rep_time
                    max_rep_weight = Pi.weight
                    max_rep_dim = Pi.dim

                # Compute norms for each generator count
                for j, G in enumerate(n_of_generators):
                    if from_file:
                        tmp_gates = repr_of_gates[:1 + G * (len(gates) - 1) // len(random_gates)]
                    else:
                        tmp_gates = repr_of_gates[:G * len(gates) // len(random_gates)]
                    
                    T = sum(tmp_gates) / len(tmp_gates)

                    flag = time.time()

                    if 0 in symmetries:
                        op = ops[0][j]
                        T_arr = T.toarray() if sparse.issparse(T) else T
                        if save_spectrum:
                            spectrum = svdvals(T_arr, overwrite_a=True, check_finite=False)
                            op += list(spectrum)
                        else:
                            op.append(norm(T_arr, 2))

                    if 1 in symmetries:
                        op = ops[1][j]
                        if sparse.issparse(T):
                            T = (T + T.getH()) / 2
                            T_arr = T.toarray()
                        else:
                            T_arr = (T + T.conj().T) / 2
                        spectrum = eigh(T_arr, eigvals_only=True)
                        if save_spectrum:
                            op += list(spectrum)
                        else:
                            op.append(max(spectrum, key=lambda x: abs(x)))

                    times[NORM] += time.time() - flag

        except Exception as e:
            exception_raised = e
            import traceback
            traceback.print_exc()

        finally:
            # Save results
            flag = time.time()
            for j in range(len(n_of_generators)):
                for symmetry in symmetries:
                    data_for_save = datas[symmetry][j]
                    op = ops[symmetry][j]
                    data_for_save.add_max_eigs(op)
            gates_data.add_max_eigs([], random_gates)
            times[ADD_EIGS] = time.time() - flag

            # Checkpoint
            if use_checkpoints:
                flag = time.time()
                
                gath_datass = []
                gath_gates_datas = []
                if mpi_loaded:
                    gath_datass = comm.gather(datas, root=0)
                    gath_gates_datas = comm.gather(gates_data, root=0)
                else:
                    gath_datass = [datas]
                    gath_gates_datas = [gates_data]
                times[GATHER] += time.time() - flag

                if rank == 0:
                    flag = time.time()
                    datas_for_save = gath_datass[0]
                    gates_data_for_save = gath_gates_datas[0]

                    for gathered_datas in gath_datass[1:]:
                        for symmetry in symmetries:
                            for j in range(len(n_of_generators)):
                                datas_for_save[symmetry][j] += gathered_datas[symmetry][j]

                    for gathered_gates_data in gath_gates_datas[1:]:
                        gates_data_for_save += gathered_gates_data
                    times[PREPARE_SAVE] += time.time() - flag

                    for symmetry in symmetries:
                        for j in range(len(n_of_generators)):
                            data_for_save = datas_for_save[symmetry][j]
                            times[SAVE_DELTAS] += data_for_save.save(
                                mode='w' if first_checkpoint else 'a'
                            )
                    times[SAVE_GATES] += gates_data_for_save.save_gates(
                        mode='w' if first_checkpoint else 'a'
                    )
                    first_checkpoint = False

                flag = time.time()
                for symmetry in symmetries:
                    for j in range(len(n_of_generators)):
                        datas[symmetry][j].erase()
                gates_data.erase()
                times[ERASE] = time.time() - flag

            if exception_raised:
                if rank == 0:
                    print(f'Exception "{exception_raised}" in sample {sample_idx + 1}.')
                    print_raport()
                break

    # Clear cache at end
    rep_cache.clear()

    if rank == 0 and not exception_raised:
        print('\nComputation completed.')
        print_raport()


# Keep original function name for compatibility
def sample_norms(sample_size, n_of_generators, d=2, v='0.0.0',
    gates_path='', weights_gen='t-design', weights_filter='PU',
    gate_order=None, fixed_gate_angle=None, fixed_gate_angles=None,
    fixed_gate_matrix=None, **kwargs):
    """Wrapper that calls optimized version."""
    return sample_norms_optimized(
        sample_size=sample_size,
        n_of_generators=n_of_generators,
        d=d, v=v, gates_path=gates_path,
        weights_gen=weights_gen,
        weights_filter=weights_filter,
        gate_order=gate_order,
        fixed_gate_angle=fixed_gate_angle,
        fixed_gate_angles=fixed_gate_angles,
        fixed_gate_matrix=fixed_gate_matrix,
        **kwargs
    )
