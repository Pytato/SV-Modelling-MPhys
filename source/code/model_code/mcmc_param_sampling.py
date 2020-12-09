from numba import njit

import numpy as np
from scipy import stats as scistats
import math


def var_eta_sample(
        h_set: np.ndarray,
        phi: float,
        mu: float
) -> float:
    n_val = len(h_set)
    a_val = (1 / 2) * (
            (1 - phi * phi) * ((h_set[0] - mu) ** 2) +
            np.sum(np.square(h_set[1:] - mu - phi * (h_set[:-1] - mu)))
    )

    return scistats.invgamma.rvs(n_val / 2.0, size=1)[0] * a_val


@njit
def mu_sample(
        h_set: np.ndarray,
        var_eta: float,
        phi: float
) -> float:
    n = len(h_set)
    b_val = (1 - phi * phi) + (n - 1) * (1 - phi) * (1 - phi)
    c_val = (1 - phi * phi) * h_set[0] + \
            (1 - phi) * np.sum(h_set[1:] - phi * h_set[:-1])

    return np.random.normal(c_val / b_val, math.sqrt(var_eta / b_val))


@njit
def __phi_2_sample_np(
        h_set: np.ndarray,
        var_eta: float,
        mu: float
) -> float:
    d_val = -1 * ((h_set[0] - mu) ** 2) + np.sum(np.square(h_set[:-1] - mu))

    e_val = np.sum((h_set - mu) * (np.append([0], h_set[:-1]) - mu))

    local_mu = e_val / d_val
    local_var = var_eta / d_val

    return_phi_val = np.random.normal(local_mu, math.sqrt(local_var), 1)[0]
    while abs(return_phi_val) >= 1:
        return_phi_val = np.random.normal(local_mu, math.sqrt(local_var), 1)[0]

    return return_phi_val

#
# phi_samples = 0
# phi_rejects = 0


@njit
def phi_sample(
        old_phi: float,
        h_set: np.ndarray,
        var_eta: float,
        mu: float
) -> float:
    """Generates a new value of phi with the metropolis hastings algorithm,
        either rejecting and passing the old phi value or accepting and
        returning the new value.
    """
    # global phi_rejects
    # global phi_samples

    # phi_samples += 1

    phi_candidate = __phi_2_sample_np(h_set, var_eta, mu)
    accept_prob = min(math.sqrt((1 - phi_candidate * phi_candidate) / (1 - old_phi * old_phi)), 1)
    mh_uniform_test_val = np.random.uniform(0.0, 1.0)

    if mh_uniform_test_val < accept_prob:
        return phi_candidate
    else:
        # phi_rejects += 1
        return old_phi


def iter_samp_param(
        old_h_set: np.ndarray,
        old_phi: float,
        old_var_eta: float,
        old_mu: float,
) -> (float, float, float):
    """"""

    new_phi = phi_sample(old_phi, old_h_set, old_var_eta, old_mu)
    new_var_eta = var_eta_sample(old_h_set, old_phi, old_mu)
    new_mu = mu_sample(old_h_set, old_var_eta, old_phi)

    return new_phi, new_var_eta, new_mu


if __name__ == "__main__":
    pass
