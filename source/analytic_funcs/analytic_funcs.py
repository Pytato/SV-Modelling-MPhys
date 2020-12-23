from typing import Union, List

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import math


@njit
def _gen_eta_t_values(eta_variance_val, n_to_gen=1):
    """Generates a set of `n_to_gen` length of eta_t values,
        uses N(0, eta_variance_val)

    :param eta_variance_val: Value for eta_t variance
    :type eta_variance_val: float
    :param n_to_gen: Length of eta_t set to generate
    :type n_to_gen: int

    :return: Set of eta_t values
    :rtype: np.ndarray(float)
    """

    return np.random.normal(0.0, math.sqrt(eta_variance_val), n_to_gen)


@njit
def expectation_val_h_t(t: Union[int, np.ndarray, List], h_1, mu, phi):
    """Expectation value for h_t, tends toward mu in the limit of t.

    :param t: Time point of h_t
    :param h_1: Initial value of h
    :type h_1: float
    :param mu: Fixed model parameter
    :type mu: float
    :param phi: Fixed model parameter
    :type phi: float
    """
    return mu + np.multiply(np.power(phi, np.subtract(t, 1)), (h_1 - mu))


@njit
def variance_h_t(t: Union[int, np.ndarray, List], eta_variance, phi):
    """Variance value for h_t, tends toward eta_variance/(1-phi**2)

    :param t: Time point of h_t
    :param eta_variance: Variance of the normal distribution used to
        generate volatility errors
    :type eta_variance: float
    :param phi: Fixed model parameter
    :type phi: float
    """

    return eta_variance * (1.0 - np.power(phi, np.multiply(2, t) - 4)) / (1 - phi*phi)


if __name__ == "__main__":
    t_set = np.arange(1, 1001)
    fig, ax = plt.subplots()
    ax.plot(t_set, expectation_val_h_t(t_set, 0.0, -1.0, 0.97))
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(t_set, variance_h_t(t_set, 0.05, 0.97))
    fig.show()
