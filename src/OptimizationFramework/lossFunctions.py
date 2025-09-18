import numpy as np

Array = np.ndarray
# ======================================================================
#                           Test functions
# ======================================================================


def rastrigin(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def sphere(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def ellipsoid(x: Array) -> float:
    """
    Ellipsoid function.
    f(x) = sum_{i=1..n} [ (10^6)^((i-1)/(n-1)) * x_i^2 ]
    A popular, strongly-conditioned test function.
    """
    x = np.array(x)
    n = len(x)
    total = 0.0
    for i in range(n):
        alpha = (10**6) ** (i / (n - 1))
        total += alpha * x[i] ** 2
    return float(total)


def rosenbrock(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e)
