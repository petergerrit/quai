# author: Piotr Dulian

from fractions import Fraction
import math
from typing import Iterable
import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import logm, expm

from gtPattern import GTPattern




def comm(a, b):
    """ Evaluates commutator of a and b. Function was implemented to work with
    scipy.sparse class.

    Args:
        a (object): object with implemented __mul__ and __sub__
        b (object): object with implemented __mul__ and __sub__

    Returns:
        object: a * b - b * a
    """
    return a * b - b * a


class uRepresentation:
    """Class computing u(n) algebra representation using Gelfand-Tsetlin construction.
    Name, against the convention, starts with small letter to highlight that this is 
    u(n) algebra representation and not U(n) group representation.  
    """

    def __init__(self, weight):
        """Class constructor. Initiates some required variables but it doesn't compute
        representation's basis.

        Args:
            weight (Iterable[int]): representation's weight
        """        
        self.weight = weight

        self.gt = GTPattern(self.weight)
        self.gt.set_to_highest()

        # Representation's dimension.
        self.dim = self.gt.get_index() + 1

        # Algebra's dimension.
        self.n = len(weight)

        # Algebra represetation's basis.
        self.e = [[[] for _ in range(self.n)] for _ in range(self.n)]


    def pattern_to_index(self, gt: GTPattern) -> int:
        return self.dim - gt.get_index() - 1


    def get_cartan_element(self, k):
        """Evaluates representation of E(k, k) or | k >< k | in braket notation.

        Args:
            k (int): 0 <= k <= self.dim - 1

        Returns:
            scipy.sparse.csr_matrix: representation of E(k, k)
        """
        self.gt.set_to_highest()
        indices = []
        data = []

        for i in range(self.dim):
            s = 0
            for j in range(k + 1):
                s += self.gt[k][j]

            for j in range(k):
                s -= self.gt[k - 1][j]

            if s != 0:
                indices.append(i)
                data.append(s)

            self.gt.one_smaller()

        return sparse.csr_matrix((data, (indices, indices)), (self.dim, self.dim))


    def get_pos_simple_root(self, k):
        """Evaluates representation of E(k - 1, k) or | k - 1 >< k | in braket notation.

        Args:
            k (int): 1 <= k <= self.dim - 1

        Returns:
            scipy.sparse.csr_matrix: representation of E(k - 1, k)
        """        
        self.gt.set_to_highest()
        row = []
        column = []
        data = []

        for l in range(self.dim):
            for j in range(k):
                m = self.gt
                x, y = 1, 1

                gt2 = self.gt.add_one(k - 1, j)
                if not gt2.correct:
                    continue

                for i in range(k + 1):
                    x *= m[k][i] - i - m[k - 1][j] + j

                for i in range(k - 1):
                    x *= m[k - 2][i] - i - m[k - 1][j] + j - 1

                for i in range(k):
                    if i == j:
                        continue
                    y *= m[k - 1][i] - i - m[k - 1][j] + j
                    y *= m[k - 1][i] - i - m[k - 1][j] + j - 1

                s = math.sqrt(- x / y)
                if s != 0:
                    row.append(self.pattern_to_index(gt2))
                    column.append(l)
                    data.append(s)

            self.gt.one_smaller()

        return sparse.csr_matrix((data, (row, column)), (self.dim, self.dim))


    def get_neg_simple_root(self, k):
        """Evaluates representation of E(k, k - 1) or | k >< k - 1 | in braket notation.

        Args:
            k (int): 1 <= k <= self.dim - 1

        Returns:
            scipy.sparse.csr_matrix: representation of E(k, k - 1)
        """        
        self.gt.set_to_highest()
        row = []
        column = []
        data = []

        for l in range(self.dim):
            for j in range(k):
                m = self.gt
                x, y = 1, 1

                gt2 = self.gt.subtract_one(k - 1, j)
                if not gt2.correct:
                    continue

                for i in range(k + 1):
                    x *= m[k][i] - i - m[k - 1][j] + j + 1

                for i in range(k - 1):
                    x *= m[k - 2][i] - i - m[k - 1][j] + j

                for i in range(k):
                    if i == j:
                        continue
                    y *= m[k - 1][i] - i - m[k - 1][j] + j + 1
                    y *= m[k - 1][i] - i - m[k - 1][j] + j

                s = math.sqrt(- x / y)
                if s != 0:
                    row.append(self.pattern_to_index(gt2))
                    column.append(l)
                    data.append(s)

            self.gt.one_smaller()

        return sparse.csr_matrix((data, (row, column)), (self.dim, self.dim))


    def make_base(self):
        """Evaluates representation ofbasis element E(j, k) and saves it in self.e[j][k]
        for all 0 <= j, k <= self.n - 1.
        """        
        self.e[0][0] = self.get_cartan_element(0)
        for i in range(1, self.n):
            self.e[i][i] = self.get_cartan_element(i)
            self.e[i][i - 1] = self.get_neg_simple_root(i)
            self.e[i - 1][i] = self.get_pos_simple_root(i)

        for i in range(2, self.n):
            for j in range(self.n - i):
                k = math.floor((i + 2 * j) / 2)
                self.e[j][i + j] = comm(self.e[j][k], self.e[k][i + j])
                self.e[i + j][j] = comm(self.e[i + j][k], self.e[k][j])


    def get_representation(self, a):
        """Computes representation of a. Should be used after evaluateBase().

        Args:
            a (numpy.array): self.n by self.n matrix

        Returns:
            scipy.sparse.csr_matrix: representation of a
        """        
        out = sparse.csr_matrix((self.dim, self.dim), dtype=complex)

        for i in range(self.n):
            for j in range(self.n):
                out += a[i][j] * self.e[i][j]

        return out


    def __call__(self, a):
        """Computes representation of a. Should be used after evaluateBase().

        Args:
            a (numpy.array): self.n by self.n matrix

        Returns:
            scipy.sparse.csr_matrix: representation of a
        """        
        return self.get_representation(a)


    @staticmethod
    def weight_to_dim(weight):
        """Computes dimension of rpresentation with given weight. It is substantially
        faster than uRepresentation(weight).dim. It implements the Weyl dimension
        formula for u(n).

        Args:
            weight (Iterable[int]): representation's weight

        Returns:
            int: representation's dimension
        """        
        nominator = 1
        denominator = 1
        for i in range(len(weight)):
            for j in range(i + 1, len(weight)):
                nominator *= (weight[i] - weight[j] + j - i)
                denominator *= (j - i)

        return int(nominator // denominator)




class suRepresentation(uRepresentation):
    """Class computing su(n) algebra representation using Gelfand-Tsetlin construction.
    Name, against the convention, starts with small letter to highlight that this is 
    su(n) algebra representation and not SU(n) group representation.  
    """

    @staticmethod
    def unspecial_weight(weight, sum_zero=False):
        """Transforms weight of a su(n) algebra representation into a weight of a u(n) 
        algebra representation which is equivalent on su(n) with the initial representation.

        Args:
            weight (Iterable[int]): su(n) representation's weight in E(i, i) - E(i + 1, i + 1)
            basis
            sum_zero (bool): whether the new weights' indices should sum to zero.

        Returns:
            weight (list[int]): u(n) representation's weight in E(i, i) basis
        """    
        uw = [sum(weight[i:]) for i in range(len(weight) + 1)]
        s = sum(uw)
        d = len(uw)

        if sum_zero:
            if sum(uw) % d == 0:
                return [x - s // d for x in uw]

            raise ValueError("Weight %s cannot be transformed to weight that sums to 0." % list(weight))

        return uw


    def __init__(self, weight):
        """Class constructor. Initiates some required variables but it doesn't compute
        representation's basis.

        Args:
            weight (Iterable[int]): representation's weight in E(i, i) - E(i + 1, i + 1)
            basis
        """        
        u_weight = self.unspecial_weight(weight)

        super().__init__(u_weight)

        self.weight = weight
        self.n = len(weight) + 1


    @staticmethod
    def weight_to_dim(weight):
        """Computes dimension of rpresentation with given weight. It is substantially
        faster than suRepresentation(weight).dim.

        Args:
            weight (Iterable[int]): representation's weight

        Returns:
            int: representation's dimension
        """        
        return uRepresentation.weight_to_dim(suRepresentation.unspecial_weight(weight))


    @staticmethod
    def is_projective(weight):
        s = 0
        for i, wi in enumerate(weight):
            s += (i + 1) * wi

        return s % (len(weight) + 1) == 0


    @staticmethod
    def is_complex(weight):
        k = len(weight)
        for i in range(k // 2):
            if weight[i] != weight[-i - 1]:
                return True
        return False


    @staticmethod
    def is_real(weight):
        d = len(weight) + 1

        if suRepresentation.is_complex(weight):
            return False

        if d % 2 == 1 or d % 4 == 0:
            return True

        if d % 4 == 2 and weight[d // 2 - 1] % 2 == 0:
            return True

        return False


    @staticmethod
    def is_quaternionic(weight):
        return not (suRepresentation.is_complex(weight) or suRepresentation.is_real(weight))


class GroupRepresentation:
    def __init__(self, algebra_rep_constr, weight):
        self.pi: uRepresentation = algebra_rep_constr(weight)
        self.pi.make_base()

        # Representation's dimension.
        self.dim = self.pi.dim

        # Group's dimension.
        self.n = self.pi.n

        self.weight = self.pi.weight


    def __call__(self, U):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="splu requires CSC matrix format")
            warnings.filterwarnings(
                "ignore",
                message="spsolve is more efficient when sparse b is in the CSC matrix format"
            )
            warnings.filterwarnings(
                "ignore",
                message="Changing the sparsity structure of a csr_matrix is expensive. "\
                "lil_matrix is more efficient."
            )
            return expm(self.pi(logm(U)))



class URepresentation(GroupRepresentation):
    """Class computing U(n) group representation.
    """    
    def __init__(self, weight):
        for i in range(len(weight) - 1):
            if weight[i] < weight[i + 1]:
                raise ValueError('Weight of unitary group has to be a sequence of non-increasing'\
                                f' integers but {weight} was given.')
        super().__init__(uRepresentation, weight)


class SURepresentation(GroupRepresentation):
    """Class computing SU(n) group representation.
    """    
    def __init__(self, weight):
        super().__init__(suRepresentation, weight)
    

def unspecial_weight(w):
    # TODO: Make this a static method in Multiplicity.
    w = np.array([Fraction(sum(w[i:])) for i in range(len(w) + 1)])
    return w - np.sum(w) / len(w)


class Multiplicity:
    def __init__(self, highest_weight):
        self.highest_weight = unspecial_weight(highest_weight)
        self.computed = {str(self.highest_weight): 1}
        self.d = len(self.highest_weight)
        self.limit = np.sum(np.abs(self.highest_weight))

        # Matrix that changes basis from {e_i} to {e_i - e_i+1} called simple roots.
        self.to_simple_roots = np.array(
            [[int(j <= i) for j in range(self.d)] for i in range(self.d - 1)]
        )


    @staticmethod
    def norm_squared(x):
        return np.inner(x, x)


    def __call__(self, weight):
        if len(weight) != self.d:
            if len(weight) != self.d - 1:
                raise ValueError(
                    f'Weight has to be of length d or d-1.\n{weight} was given and d={self.d}'
                )

            weight = unspecial_weight(weight)

        if self.d == 2:
            return int(round(
                self.highest_weight[0] - self.highest_weight[1] - weight[0] + weight[1]) % 2 == 0
            )

        weight_key = str(weight)
        if weight_key in self.computed:
            return self.computed[weight_key]

        space = self.to_simple_roots @ (self.highest_weight - weight)

        if (space < 0).any(): # Checks if weight is lower than the highest weight.
            return 0

        if (np.vectorize(round)(space) != space).any(): # Checks if space is in the root lattice.
            return 0

        s = 0
        weyl_vector = np.zeros(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                root = np.zeros(self.d, dtype=int)
                root[i] = 1
                root[j] = -1

                weyl_vector += root
                # tmp = weight.copy() + k * 
                for k in range(round(min(space[i:j])), 0, -1):
                    tmp = weight + k * root

                    if np.sum(np.abs(tmp)) > self.limit:
                        continue

                    s += np.inner(root, tmp) * self(tmp) # Max depth is d(d-1)/2.

        weyl_vector /= 2
        s *= 2

        if (
            x := self.norm_squared(self.highest_weight + weyl_vector)
            - self.norm_squared(weight + weyl_vector)
        ):
            s /= x
        else:
            s = 0

        s = np.around(s)
        self.computed[weight_key] = s

        return s
