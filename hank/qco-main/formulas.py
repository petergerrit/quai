from math import sqrt, pi
import numpy as np




def km_distribution(x, S, cumulative=False):
    """Kesten-McKay distribution.

    f(x) = (S / (2pi)) * sqrt(l**2 - x**2) / (1 - x**2)
    for l = 2 * sqrt(S - 1) / S

    Args:
        x (array): arguments.
        S (int): parameter.
        cumulative (bool, optional): Whether the distribution is cumulative. Defaults to False.

    Returns:
        array or scalar: value of distribution at x.
    """
    l = 2 * sqrt(S - 1) / S
    x = np.where(-l <= x, x, -l * np.ones(len(x)))
    x = np.where(x <= l, x, l * np.ones(len(x)))

    if cumulative:
        def cdf(x0):
            mul_factor = S / (2 * pi)
            y = sqrt(1 - l**2) * np.arctan(np.sqrt(l**2 - x0**2) / (sqrt(1 - l**2) * x0))
            y += np.arcsin(x0 / l)
            return mul_factor * y

        result = np.zeros(len(x))

        # cdf has dicontinuty at 0
        neg = np.where(x < 0)
        pos = np.where(x > 0)

        result[neg] = cdf(x[neg]) + S / 4
        result[pos] = cdf(x[pos]) - (S - 2) / 4 + 1 / 2

        if 0 in x:
            result[np.where(x == 0)] = 1 / 2

        return result

    return S * np.sqrt(l**2 - x**2) / (2 * pi * (1 - x**2))


def mb_distribution(x, m=1, T=300, k=1):
    kT = k * T
    x2 = x**2
    return (m / (2 * pi * kT))**(3 / 2) * 4 * pi * x2 * np.exp(-m * x2 / (2 *kT))
