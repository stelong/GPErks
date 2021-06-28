import numpy as np

# TODO: make this a class that for a given function name returns both the function AND the input dimension


def forrester(x):
    """
    Dimensions: 1
    Interval: x in [0, 1]
    """
    return np.power(6 * x - 2, 2) * np.sin(12 * x - 4)


def currin_exp(x):
    """
    Dimensions: 2
    Interval: x_i in [0, 1] for i=1,2
    """
    return (
        (1 - np.exp(np.power(-2 * x[1], -1)))
        * (2300 * x[0] ** 3 + 1900 * x[0] ** 2 + 2092 * x[0] + 60)
        / (100 * x[0] ** 3 + 500 * x[0] ** 2 + 4 * x[0] + 20)
    )


def lim_poly(x):
    """
    Dimensions: 2
    Interval: x_i in [0, 1] for i=1,2
    """
    return (
        9
        + 2.5 * x[0]
        - 17.5 * x[1]
        + 2.5 * x[0] * x[1]
        + 19 * x[1] ** 2
        - 7.5 * x[0] ** 3
        - 2.5 * x[0] * x[1] ** 2
        - 5.5 * x[1] ** 4
        + x[0] ** 3 * x[1] ** 2
    )


def branin_rescaled(x):
    """
    Dimensions: 2
    Interval: x_i in [0, 1] for i=1,2
    """
    x0, x1 = 15 * x[0] - 5, 15 * x[1]
    return (
        np.power(x1 - 1.275 * x0 ** 2 / np.pi ** 2 + 5 * x0 / np.pi - 6, 2)
        + 10 * (1 - 0.125 / np.pi) * np.cos(x0)
        - 44.81
    ) / 51.95
