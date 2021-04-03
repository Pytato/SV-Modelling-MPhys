from numba import njit

import numpy as np
from scipy import stats as scistats
import math


def eta_var_sample(
        a_val: float,
        n_val: int,
) -> float:
    return scistats.invgamma.rvs(n_val / 2.0, size=1)[0] * a_val


@njit
def njit_eta_var_sample(
        a_val: float,
        n_val: int
) -> float:
    return 1./np.random.gamma(n_val/2., 1./a_val)


@njit
def mu_sample(
        b_val: float,
        c_val: float,
        var_eta: float
) -> float:
    return np.random.normal(c_val / b_val, math.sqrt(var_eta / b_val))


@njit
def phi_2_sample(
        d_val: float,
        e_val: float,
        var_eta: float
) -> float:

    local_mu = e_val / d_val
    local_var = var_eta / d_val
    return_phi_val = np.random.normal(local_mu, math.sqrt(local_var), 1)[0]
    while abs(return_phi_val) >= 1:
        return_phi_val = np.random.normal(local_mu, math.sqrt(local_var), 1)[0]

    return return_phi_val


@njit
def phi_sample(
        old_phi: float,
        d_val: float,
        e_val: float,
        var_eta: float
) -> float:

    phi_candidate = phi_2_sample(d_val, e_val, var_eta)
    accept_prob = min(math.sqrt((1 - phi_candidate * phi_candidate) / (1 - old_phi * old_phi)), 1)
    mh_uniform_test_val = np.random.uniform(0.0, 1.0)
    if mh_uniform_test_val <= accept_prob:
        return phi_candidate
    else:
        return old_phi


if __name__ == "__main__":
    pass
