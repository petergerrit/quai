from math import sqrt

import numpy as np
from numpy import trace, around




def flatten(l):
    return (item for sublist in l for item in sublist)


def transpose(m):
    height = len(m)
    width = len(m[0])

    t = [[] for _ in range(width)]

    for i in range(width):
        for j in range(height):
            t[i].append(m[j][i])

    return t


def list_to_string(l, p='+'):
    s = str(l[0])
    for x in l[1:]:
        s += p + str(x)
    return s


def norm2(weight):
    return sqrt(sum(w**2 for w in weight))


def in_interval(a, interval, sharp=True):
    mn, mx = interval
    return (mn < a) & (a < mx) if sharp else (mn <= a) & (a <= mx)


def signed_permutations(xs):
    if len(xs) <= 1:
        yield xs, 1
    else:
        for permutation, sign in signed_permutations(xs[1:]):
            sign *= -1
            for i in range(len(xs)):
                sign *= -1
                yield permutation[:i] + xs[0:1] + permutation[i:], sign


def hs_inner(a, b):
    return trace(a.conjugate().T @ b)


def inner_to_norm(inner):
    # abs is to avoid raising of ComplexWarning conserning casting complex to real.
    return lambda x: sqrt(abs(inner(x, x)))


def hs_norm(a):
    return inner_to_norm(hs_inner)(a)


def is_proportional(x, y, inner=hs_inner, epsilon=1e-10):
    norm = inner_to_norm(inner)

    norm_x = norm(x)
    norm_y = norm(y)
    inner_xy = inner(x, y)
    
    # If x is proportional to y then they saturate Cauchy-Schwartz inequality.
    return abs(abs(inner_xy) - norm_x * norm_y) < epsilon


def nparray_eq(array0, array1, decimals=10):
    return (around(array0 - array1, decimals) == 0).all()


def factors(n):    # (cf. https://stackoverflow.com/a/15703327/849891)
    j = 2
    while n > 1:
        for i in range(j, int(sqrt(n + 0.05)) + 1):
            if n % i == 0:
                n //= i
                j = i
                yield i
                break
        else:
            if n > 1:
                yield n
                break


def cmp(x, y):
    if x > y: return 1
    if x < y: return -1
    return 0


def hc(x: np.ndarray) -> np.ndarray:
    return x.conjugate().T
