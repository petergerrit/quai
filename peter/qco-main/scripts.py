from __future__ import annotations

from functools import cmp_to_key
import math
import random
import signal
import sys
import time

import numpy as np
from scipy.linalg import eigh, svdvals, norm
from scipy import sparse

from consts import *
from dataStructures import *
from representation import suRepresentation, SURepresentation
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
    """Returns Haar random matrix from SU(n).

    Args:
        n (int): group dimension
        rank (int): rank of the result, U**rank is proportional to Id

    Returns:
        numpy.matrix: matrix of shape (n, n)
    """
    if order is not None:
        if n != 2:
            raise NotImplementedError()

        v = get_random_SU(n, order=None)

        ks = []
        for k in range(1, order):
            if np.gcd(order, k) == 1:
                ks.append(k)
        k = random.choice(ks)
        phase = 1j * 2*np.pi * k / order
        d = np.diag([np.exp(phase), 1])
        d = np.matrix(d)
        v_dagger = np.conjugate(np.transpose(v))

        return v * d * v_dagger
        
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
        n = int(lines[0][0])
        for i in range(1, len(lines), 1):
            line = lines[i].split()

            if i % n == 1:
                gates.append([])

            gates[-1].append([complex(x) for x in line])

    return n, [np.array(x) for x in gates]


def norm_dim(weight):
    return suRepresentation.weight_to_dim(weight)


def get_SU_weights(d, max_norm, norm=norm2):
    """
    Generates all SU rep. weights with the norm smaller than J.

    Parameters
    ----------
    d : int
        Group's dimension.
    J : numerical
        Upper bound on weight's norm.
    weight_norm : function
        Function calculating weight's norm.

    Returns
    -------
    weights : list of numpy.array
        List of weights.

    """

    # Generates the first weight.
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
        raise ValueError('Weights %s and %s cannot be comapred by t.' % (w1, w2))

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


def sample_norms(sample_size, n_of_generators, d=2, v='0.0.0',
    gates_path='', weights_gen='t-design', weights_filter='PU',
    gate_order: int | None = None, **kwargs):
    comm = None
    rank = 0
    size = 1

    if mpi_loaded:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    exception_raised = None

    # Sample size.
    sample_size = int(sample_size)

    # G is the number of gates.
    Gs_slice = [int(x) for x in n_of_generators.split('-')]
    if len(Gs_slice) > 1:
        n_of_generators = range(*Gs_slice)
    else:
        n_of_generators = Gs_slice

    # Dimension of SU.
    d = int(d)

    # Is set of gates symmetric?
    symmetries = [0]
    if 'symmetric' in kwargs:
        symmetries = [1]
    if 'unsymmetric' in kwargs:
        symmetries = [0]

    save_spectrum = 'save_spectrum' in kwargs

    # Version
    v = v

    use_checkpoints = True
    first_checkpoint = True

    # Path to the file with gates.
    from_file = bool(gates_path)
    f_gates = []
    if from_file:
        d, f_gates = get_gates_from_file(gates_path)


    start = time.time()
    NORM = 'norm'
    REP = 'rep'
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
        print("Average times: ")
        max_key_length = max(len(key) for key in times)
        max_time_length = max(map(lambda x: len(str(x)), times.values()))

        def add_spaces(s): return ' - ' + s + ':' + ' ' * (max_key_length + 1 - len(s))

        for key in times:
            print(add_spaces(key) + str(round(times[key] / my_N, 4)) + 's')
        print('-' * (max_key_length + max_time_length + 10) + '\nRatios: ')
        for key in times:
            print(add_spaces(key) + str(round(times[key] / times[ALL] * 100, 2)) + '%')
        print('-' * (max_key_length + max_time_length + 10))
        print('Max representation:')
        print(add_spaces('time') + str(round(max_rep_time, 2)) + 's')
        print(add_spaces('weight') + str(max_rep_weight))
        print(add_spaces('dimension') + str(max_rep_dim))
        print('-' * (max_key_length + max_time_length + 10))
        print('It took: ' + str(times[ALL]) + 's')

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

    # The norm for weight [mu_1, ..., mu_n] is the same as for [mu_n, ..., mu_1].
    weights = filter_mirror_weights(weights)

    options = {
        **options,
        'N': int(round(math.log10(sample_size))),
        'f': 1 if from_file else 0,
        'v': v
    }

    if gate_order is not None:
        gate_order = int(gate_order)
        options = {**options, 'r': gate_order}

    if save_spectrum:
        options = {**options, 'spec': ''}

    if rank == 0:
        print("Start.")
        print("Number of processors: " + str(size))
        print("Chosen options:")
        for k in options.keys():
            print(str(k) + ": " + str(options[k]))
        print("Symmetries: " + str(symmetries))

    if mpi_loaded:
        weights = comm.scatter(divide_list(weights, size), root=0)

    representations = [SURepresentation(weight) for weight in weights]

    if mpi_loaded:
        representations = comm.gather(representations, root=0)

        if rank == 0:
            representations = list(flatten(representations))
            representations.sort(
                key=cmp_to_key(
                    lambda Pi1, Pi2: compare_weights(
                        Pi1.weight, Pi2.weight,
                        cmp_by_t=weights_gen == 't-design'
                    )
                )
            )

        representations = comm.bcast(representations, root=0)

    if rank == 0:
        print("Broadcast test.")
        print("I have " + str(len(representations)) + " representations.")

    rest = sample_size % size
    my_N = (sample_size - rest) // size
    my_N += int(rank < rest)

    datas = {}
    for symmetry in symmetries:
        tmp_options = {'s': symmetry, **options}
        datas[symmetry]: dict[int, list[QcoData]] = [
            QcoData(d, G, tmp_options, representations, rank) for G in n_of_generators
        ]

    gates_data = QcoData(d, 1, options, [], rank)

    for i in range(my_N):
        ops = {s: [[] for _ in n_of_generators] for s in symmetries}
        gates = []
        Gmax = max(n_of_generators)
        random_gates = [get_random_SU(d, gate_order) for _ in range(Gmax)]

        if from_file:
            gates = [g @ rg @ g.conjugate().T for rg in random_gates for g in f_gates]
        else:
            gates = random_gates

        try:
            for Pi in representations:
                flag = time.time()
                repr_of_gates = [Pi(gate) for gate in gates]
                rep_time = time.time() - flag
                times[REP] += rep_time

                if rep_time > max_rep_time:
                    max_rep_time = rep_time
                    max_rep_weight = Pi.weight
                    max_rep_dim = Pi.dim

                for j, G in enumerate(n_of_generators):
                    if from_file:
                        tmp_gates = repr_of_gates[:1 + G * (len(gates) - 1) // len(random_gates)]
                    else:
                        tmp_gates = repr_of_gates[:G * len(gates) // len(random_gates)]
                    T: sparse.csr_matrix = sum(tmp_gates) / len(tmp_gates)

                    flag = time.time()

                    if 0 in symmetries:
                        op = ops[0][j]
                        T_arr = T.toarray()
                        if save_spectrum:
                            spectrum = svdvals(
                                T_arr, overwrite_a=True, check_finite=False
                            )
                            op += list(spectrum)
                        else:
                            op.append(norm(T_arr, 2))

                    if 1 in symmetries:
                        op = ops[1][j]
                        T = (T + T.getH()) / 2
                        spectrum = eigh(T.toarray(), eigvals_only=True)

                        # Saving the whole spectrum makes sense only if the set is symmetric.
                        # In non-symmetric case norm of T may be different from the largest
                        # eigenvalue.
                        if save_spectrum:
                            op += list(spectrum)
                        else:
                            op.append(max(spectrum, key=lambda x: abs(x)))

                    times[NORM] += time.time() - flag

        except Exception as e:
            exception_raised = e

        finally:
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

                gath_datass: list[dict[int, list[QcoData]]] = []
                gath_gates_datas: list[QcoData] = []
                if mpi_loaded:
                    gath_datass = comm.gather(datas, root=0)
                    gath_gates_datas = comm.gather(gates_data, root=0)
                else:
                    gath_datass = [datas]
                    gath_gates_datas = [gates_data]
                times[GATHER] += time.time() - flag

                if rank == 0:
                    datass_ranks = [gath_datas[symmetries[0]][0].rank for gath_datas in gath_datass]
                    gates_ranks = [gath_gates_data.rank for gath_gates_data in gath_gates_datas]
                    if datass_ranks != gates_ranks:
                        print('Different rank order.')
                        gath_datass = sorted(gath_datass, key=lambda ds: ds[symmetries[0]][0].rank)
                        gath_gates_datas = sorted(gath_gates_datas, key=lambda g: g.rank)

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
                    print('Checkpoint achieved', file=sys.stderr)

                flag = time.time()
                for symmetry in symmetries:
                    for j in range(len(n_of_generators)):
                        datas[symmetry][j].erase()
                gates_data.erase()
                times[ERASE] = time.time() - flag

            if exception_raised:
                if rank == 0:
                    print(f'Exception "{exception_raised}" in the {i+1}-th iteration.')
                    print_raport()
            
                break

    if rank == 0 and not exception_raised:
        print('Computation completed.')
        print_raport()


def get_partitions(n, k, bound=math.inf):
    """Returns all partitions of n into k terms non-grater than bound.

    The partition of the integer n of length k is a list of non-negative
    integers of length k sorted in descending order such that: 
        partition[0] + ... + partition[k - 1] = n.

    Args:
        n (int): number to be partitioned.
        k (int): maximal number of terms in partition.
        bound (numeric, optional): Upper bound for terms in partition. Defaults to math.inf.

    Returns:
        [[int]]: list of partitions.
    """
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
        raise ValueError('Partitions have to be of the same legth.')

    return [partition1[i] - partition2[- i - 1] for i in range(len(partition1))]


def t_design_weights(d, t):
    """Returns all SU(d) represetnation weights appearing in the decomposition of t-design.

    Args:
        d (int): group dimension
        t (int): parameter of t-design

    Returns:
        [(int)]: list of tuples represeting weights.
    """

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
