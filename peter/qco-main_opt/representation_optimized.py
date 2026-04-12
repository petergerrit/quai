"""
Optimized representation module for u(n) and su(n) algebras.

Key optimizations:
1. Pre-compute all GT patterns and transitions upfront
2. Vectorized operations using numpy
3. Parallel representation construction using multiprocessing
4. Avoid repeated deepcopy operations
"""
from __future__ import annotations

import math
import warnings
from typing import Iterable, Optional, Tuple, Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from scipy import sparse
from scipy.linalg import logm, expm
from scipy.sparse.linalg import expm as sparse_expm

from gtPattern_optimized import precompute_gt_data


def _get_row_value(pattern: tuple, n: int, level: int, pos: int) -> int:
    """Get value at row 'level', position 'pos' from pattern tuple."""
    row_idx = n - 1 - level
    if row_idx < 0 or row_idx >= len(pattern):
        return 0
    row = pattern[row_idx]
    if pos < 0 or pos >= len(row):
        return 0
    return row[pos]


class uRepresentationOptimized:
    """Optimized u(n) algebra representation using pre-computed GT patterns."""

    def __init__(self, weight: Iterable[int], precomputed: Optional[Dict] = None):
        """
        Initialize the representation.
        
        Args:
            weight: The highest weight (top row of GT pattern)
            precomputed: Optional pre-computed GT data (for parallel construction)
        """
        self.weight = tuple(weight)
        self.n = len(weight)
        
        if precomputed is not None:
            self._gt_data = precomputed
        else:
            self._gt_data = precompute_gt_data(self.weight)
        
        self.dim = self._gt_data['dim']
        self.patterns = self._gt_data['patterns']
        self.pattern_to_idx = self._gt_data['pattern_to_idx']
        self.transitions_add = self._gt_data['transitions_add']
        self.transitions_sub = self._gt_data['transitions_sub']
        
        # Algebra representation basis
        self.e = [[None for _ in range(self.n)] for _ in range(self.n)]
        self._base_computed = False

    def get_cartan_element(self, k: int) -> sparse.csr_matrix:
        """Compute representation of E(k, k)."""
        indices = []
        data = []
        
        for idx, pattern in enumerate(self.patterns):
            # Sum over entries in row k
            s = 0
            for j in range(k + 1):
                s += _get_row_value(pattern, self.n, k, j)
            # Subtract entries from row k-1
            for j in range(k):
                s -= _get_row_value(pattern, self.n, k - 1, j)
            
            if s != 0:
                indices.append(idx)
                data.append(s)
        
        return sparse.csr_matrix((data, (indices, indices)), (self.dim, self.dim))

    def get_pos_simple_root(self, k: int) -> sparse.csr_matrix:
        """Compute representation of E(k-1, k) - raising operator."""
        rows = []
        cols = []
        data = []
        
        for idx, pattern in enumerate(self.patterns):
            for j in range(k):
                # Check if transition is valid
                new_idx = self.transitions_add.get((idx, k - 1, j), -1)
                if new_idx < 0:
                    continue
                
                # Compute the coefficient
                x, y = 1.0, 1.0
                
                # Get values from pattern
                m_k = [_get_row_value(pattern, self.n, k, i) for i in range(k + 1)]
                m_k_1 = [_get_row_value(pattern, self.n, k - 1, i) for i in range(k)]
                m_k_2 = [_get_row_value(pattern, self.n, k - 2, i) for i in range(k - 1)] if k >= 2 else []
                
                for i in range(k + 1):
                    x *= m_k[i] - i - m_k_1[j] + j
                
                for i in range(k - 1):
                    x *= m_k_2[i] - i - m_k_1[j] + j - 1
                
                for i in range(k):
                    if i == j:
                        continue
                    diff = m_k_1[i] - i - m_k_1[j] + j
                    y *= diff * (diff - 1)
                
                if y == 0:
                    continue
                
                s = math.sqrt(-x / y)
                if s != 0:
                    rows.append(new_idx)
                    cols.append(idx)
                    data.append(s)
        
        return sparse.csr_matrix((data, (rows, cols)), (self.dim, self.dim))

    def get_neg_simple_root(self, k: int) -> sparse.csr_matrix:
        """Compute representation of E(k, k-1) - lowering operator."""
        rows = []
        cols = []
        data = []
        
        for idx, pattern in enumerate(self.patterns):
            for j in range(k):
                # Check if transition is valid
                new_idx = self.transitions_sub.get((idx, k - 1, j), -1)
                if new_idx < 0:
                    continue
                
                # Compute the coefficient
                x, y = 1.0, 1.0
                
                # Get values from pattern
                m_k = [_get_row_value(pattern, self.n, k, i) for i in range(k + 1)]
                m_k_1 = [_get_row_value(pattern, self.n, k - 1, i) for i in range(k)]
                m_k_2 = [_get_row_value(pattern, self.n, k - 2, i) for i in range(k - 1)] if k >= 2 else []
                
                for i in range(k + 1):
                    x *= m_k[i] - i - m_k_1[j] + j + 1
                
                for i in range(k - 1):
                    x *= m_k_2[i] - i - m_k_1[j] + j
                
                for i in range(k):
                    if i == j:
                        continue
                    diff = m_k_1[i] - i - m_k_1[j] + j
                    y *= (diff + 1) * diff
                
                if y == 0:
                    continue
                
                s = math.sqrt(-x / y)
                if s != 0:
                    rows.append(new_idx)
                    cols.append(idx)
                    data.append(s)
        
        return sparse.csr_matrix((data, (rows, cols)), (self.dim, self.dim))

    def make_base(self):
        """Compute all basis elements E(i,j)."""
        if self._base_computed:
            return
        
        # Diagonal elements (Cartan subalgebra)
        for i in range(self.n):
            self.e[i][i] = self.get_cartan_element(i)
        
        # Simple roots
        for i in range(1, self.n):
            self.e[i][i - 1] = self.get_neg_simple_root(i)
            self.e[i - 1][i] = self.get_pos_simple_root(i)
        
        # Higher roots via commutators
        for delta in range(2, self.n):
            for j in range(self.n - delta):
                k = (delta + 2 * j) // 2
                # E[j, j+delta] = [E[j,k], E[k, j+delta]]
                self.e[j][j + delta] = self._comm(self.e[j][k], self.e[k][j + delta])
                self.e[j + delta][j] = self._comm(self.e[j + delta][k], self.e[k][j])
        
        self._base_computed = True

    @staticmethod
    def _comm(a, b):
        """Compute commutator [a, b] = ab - ba."""
        return a @ b - b @ a

    def get_representation(self, a: np.ndarray) -> sparse.csr_matrix:
        """Compute representation of algebra element a."""
        if not self._base_computed:
            self.make_base()
        
        rows, cols, vals = [], [], []
        for i in range(self.n):
            for j in range(self.n):
                coeff = a[i, j]
                if coeff == 0:
                    continue
                coo = self.e[i][j].tocoo()
                rows.append(coo.row)
                cols.append(coo.col)
                vals.append(coeff * coo.data)
        
        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)
        else:
            rows = cols = vals = []
        
        return sparse.csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim), dtype=complex)

    def __call__(self, a: np.ndarray) -> sparse.csr_matrix:
        return self.get_representation(a)

    @staticmethod
    def weight_to_dim(weight: Iterable[int]) -> int:
        """Compute dimension using Weyl formula (no pattern enumeration needed)."""
        weight = list(weight)
        n = len(weight)
        num = 1
        den = 1
        for i in range(n):
            for j in range(i + 1, n):
                num *= (weight[i] - weight[j] + j - i)
                den *= (j - i)
        return num // den


class suRepresentationOptimized(uRepresentationOptimized):
    """Optimized su(n) algebra representation."""

    @staticmethod
    def unspecial_weight(weight: Iterable[int], sum_zero: bool = False) -> list:
        """Convert su(n) weight to u(n) weight."""
        weight = list(weight)
        uw = [sum(weight[i:]) for i in range(len(weight) + 1)]
        s = sum(uw)
        d = len(uw)
        
        if sum_zero:
            if s % d == 0:
                return [x - s // d for x in uw]
            raise ValueError(f"Weight {weight} cannot be transformed to sum-zero weight.")
        
        return uw

    def __init__(self, weight: Iterable[int], precomputed: Optional[Dict] = None):
        u_weight = self.unspecial_weight(weight)
        super().__init__(u_weight, precomputed)
        self.weight = tuple(weight)
        self.n = len(weight) + 1

    @staticmethod
    def weight_to_dim(weight: Iterable[int]) -> int:
        return uRepresentationOptimized.weight_to_dim(suRepresentationOptimized.unspecial_weight(weight))

    @staticmethod
    def is_projective(weight: Iterable[int]) -> bool:
        weight = list(weight)
        s = sum((i + 1) * w for i, w in enumerate(weight))
        return s % (len(weight) + 1) == 0

    @staticmethod
    def is_complex(weight: Iterable[int]) -> bool:
        weight = list(weight)
        k = len(weight)
        for i in range(k // 2):
            if weight[i] != weight[-i - 1]:
                return True
        return False

    @staticmethod
    def is_real(weight: Iterable[int]) -> bool:
        weight = list(weight)
        d = len(weight) + 1
        if suRepresentationOptimized.is_complex(weight):
            return False
        if d % 2 == 1 or d % 4 == 0:
            return True
        if d % 4 == 2 and weight[d // 2 - 1] % 2 == 0:
            return True
        return False

    @staticmethod
    def is_quaternionic(weight: Iterable[int]) -> bool:
        return not (suRepresentationOptimized.is_complex(weight) or 
                   suRepresentationOptimized.is_real(weight))


def _build_single_representation(args):
    """Worker function for parallel representation construction."""
    weight, weight_class = args
    if weight_class == 'su':
        rep = SURepresentationOptimized(weight)
    else:
        rep = URepresentationOptimized(weight)
    return rep


class GroupRepresentationOptimized:
    """Optimized group representation with parallel construction support."""

    def __init__(self, algebra_rep_constr, weight: Iterable[int]):
        self.pi = algebra_rep_constr(weight)
        self.pi.make_base()
        self.dim = self.pi.dim
        self.n = self.pi.n
        self.weight = self.pi.weight

    def __call__(self, U: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pi_log = self.pi(logm(U))
            if sparse.issparse(pi_log):
                return sparse_expm(pi_log.tocsc())
            return expm(pi_log)


class URepresentationOptimized(GroupRepresentationOptimized):
    """U(n) group representation."""
    def __init__(self, weight: Iterable[int]):
        super().__init__(uRepresentationOptimized, weight)


class SURepresentationOptimized(GroupRepresentationOptimized):
    """SU(n) group representation."""
    def __init__(self, weight: Iterable[int]):
        super().__init__(suRepresentationOptimized, weight)


def build_representations_parallel(weights: List, weight_class: str = 'su', 
                                   n_workers: Optional[int] = None) -> List:
    """
    Build multiple representations in parallel.
    
    Args:
        weights: List of weights to build representations for
        weight_class: 'su' for SU(n) or 'u' for U(n)
        n_workers: Number of parallel workers (default: cpu_count)
    
    Returns:
        List of built representations
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    args = [(tuple(w), weight_class) for w in weights]
    
    with Pool(n_workers) as pool:
        representations = pool.map(_build_single_representation, args)
    
    return representations


# Compatibility layer - drop-in replacements
suRepresentation = suRepresentationOptimized
uRepresentation = uRepresentationOptimized  
SURepresentation = SURepresentationOptimized
URepresentation = URepresentationOptimized
