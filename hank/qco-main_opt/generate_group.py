from collections import deque
from itertools import product

import numpy as np




EPS = 1e-10


def is_proportional(x: np.ndarray, y: np.ndarray) -> bool:
    if np.all(np.abs(x) < EPS) or np.all(np.abs(y) < EPS):
        return True

    shape = x.shape
    for it in product(*(range(n) for n in shape)):
        if np.abs(x[it]) > EPS:
            factor = y[it] / x[it]
    return np.all(np.abs(x * factor - y) < EPS)


def from_generators(generators: list[np.ndarray]) -> list[np.ndarray]:
    gen_group = [np.identity(generators[0].shape[0])]
    q = deque(generators)
    while q:
        g = q.popleft()
        for h in gen_group:
            gh = g @ h
            not_in = True
            for i in gen_group:
                if is_proportional(gh, i):
                    not_in = False
                    break
            if not_in:
                gen_group.append(gh)
                q.append(gh)

    return list(gen_group)
