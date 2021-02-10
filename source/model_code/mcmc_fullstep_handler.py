from source.model_code import hmc_sampling, mcmc_param_sampling

from numba import njit
from typing import Optional

import numpy as np
import math


def _get_accepted_trajectory(
        h_init: np.ndarray,
        y_series: np.ndarray,
        phi_input: float,
        mu_input: float,
        var_eta_input: float,
        trajectory_length=1.,
        n_steps=40,
        max_attempts=200
) -> [np.ndarray, np.ndarray, int]:
    attempts_made = 0
    while attempts_made < max_attempts:
        attempts_made += 1
        p_init = np.random.normal(0, 2, len(h_init))
        h_candidate, p_candidate = hmc_sampling.integrate_trajectory(
            h_init, p_init, trajectory_length, n_steps, y_series,
            phi_input, mu_input, var_eta_input
        )
        hamiltonian_delta = hmc_sampling.hamiltonian(
            h_candidate, p_candidate, y_series, phi_input, mu_input, var_eta_input
        ) - hmc_sampling.hamiltonian(
            h_init, p_init, y_series, phi_input, mu_input, var_eta_input
        )
        # print(hamiltonian_delta)
        if np.random.uniform(0, 1) < np.min([np.exp(-hamiltonian_delta), 1.0]):
            break
        # try:
        #     phi_input, mu_input, var_eta_input = mcmc_param_sampling.iter_samp_param(h_init, phi_input,
        #                                                                              mu_input, var_eta_input)
        # except:
        #     pass

    return h_candidate, p_candidate, attempts_made


def step_mcmc(
        h_init: np.ndarray,
        y_series: np.ndarray,
        phi_input: float,
        mu_input: float,
        var_eta_input: float,
        trajectory_length=1.,
        n_steps=40,
        max_attempts=200
) -> (np.ndarray, (float, float, float), Optional[int]):
    h_out, p_out, attempts_made = _get_accepted_trajectory(
        h_init, y_series, phi_input, mu_input, var_eta_input, trajectory_length=trajectory_length,
        n_steps=n_steps, max_attempts=max_attempts
    )
    if attempts_made == max_attempts:
        print("Inference failed")
        return h_init, (phi_input, mu_input, var_eta_input), attempts_made

    new_phi, new_mu, new_var_eta = mcmc_param_sampling.iter_samp_param(h_out, phi_input,
                                                                       mu_input, var_eta_input)

    return h_out, (new_phi, new_mu, new_var_eta), attempts_made


if __name__ == "__main__":
    pass
