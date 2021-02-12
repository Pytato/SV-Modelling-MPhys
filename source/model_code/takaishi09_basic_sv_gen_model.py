import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import math


@njit
def gen_eta_t_values(eta_variance_val, n_to_gen=1):
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
def gen_epsilon_value_set(n_to_gen=1):
    """Generates a set of `n_to_gen` length of epsilon_t values,
        uses N(0, 1).

    :param n_to_gen: Length of epsilon_t set to generate
    :type n_to_gen: int

    :return: Set of epsilon_t values
    :rtype: np.ndarray(float)
    """

    return np.random.normal(0.0, 1.0, n_to_gen)


@njit
def _gen_h_t_val(h_t_minus_1, mu, phi, eta_t):
    """Steps to h_t from h_{t-1}

    :param h_t_minus_1: h for prior time-step
    :type h_t_minus_1: float
    :param mu: Fixed model parameter
    :type mu: float
    :param phi: Fixed model parameter
    :type phi: float
    :param eta_t: volatility variance random "error" value
    :type eta_t: float

    :return: h_t value for current time-step
    :rtype: float
    """

    return mu + phi*(h_t_minus_1 - mu) + eta_t


@njit
def gen_h_t_set(h_initial, mu, phi, eta_t_set):
    """Generates set of h_t based on length of eta_t_set

    :param h_initial: Initial h_t value to iterate from
    :type h_initial: float
    :param mu: Fixed model parameter
    :type mu: float
    :param phi: Fixed model parameter
    :type phi: float
    :param eta_t_set: List of pre-generated eta_t values
    :type eta_t_set: np.ndarray[float]

    :return: Set of length {len(eta_t_set) + 1} h_t values
    :rtype: np.ndarray[float]
    """

    h_val_set = np.zeros(len(eta_t_set)+1, dtype=np.double)
    h_val_set[0] = h_initial

    for i in range(1, len(h_val_set)):
        h_val_set[i] = _gen_h_t_val(h_val_set[i-1], mu, phi, eta_t_set[i-1])

    return h_val_set[1:]


@njit
def gen_y_t_vals(phi, mu, eta_variance, num_steps_to_gen=1000):
    """Generates set of `num_steps_to_gen` length time series data
            representing the pricing of a stock option

    :param phi: Fixed parameter, represents level of auto-correlation of the option pricing,
    i.e. how closely related volatility is to previous values of volatility
    :type phi: float
    :param mu: Fixed parameter across model
    :type mu: float
    :param eta_variance: Initial volatility internal random variance, used
    to inform the model's volatility generation method.
    :type eta_variance: float
    :param num_steps_to_gen: Number of time steps to generate
    :type num_steps_to_gen: int

    :return: Array of time-series y_t data
    :rtype: np.ndarray(np.double)
    """

    epsilon_t_set = gen_epsilon_value_set(n_to_gen=num_steps_to_gen)
    eta_t_set = gen_eta_t_values(eta_variance, n_to_gen=num_steps_to_gen)
    h_1 = 0.0
    h_val_set = gen_h_t_set(h_1, mu, phi, eta_t_set)
    y_val_set = np.multiply(np.exp(h_val_set/2.0), epsilon_t_set)

    return y_val_set, h_val_set


if __name__ == "__main__":
    print("Test _gen_epsilon_value_set and _gen_eta_t_values")
    print(gen_epsilon_value_set(n_to_gen=100))
    print(gen_eta_t_values(0.05, n_to_gen=100))

    print("Test _gen_t_t_vals")
    y_t_val_test = gen_y_t_vals(0.97, -1.0, 0.05, num_steps_to_gen=5000)
    fig, ax = plt.subplots()
    ax.plot(y_t_val_test)
    fig.show()
