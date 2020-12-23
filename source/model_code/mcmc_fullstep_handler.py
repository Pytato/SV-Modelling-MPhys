from source.model_code import hmc_sampling

import numpy as np
import math


def _get_accept_trajectory(
        h_init: np.ndarray,
        y_series: np.ndarray,
        phi_input: float,
        mu_input: float,
        var_eta_input: float,
        trajectory_length=1.,
        n_steps=40,
        max_attempts=200
) -> (np.ndarray, np.ndarray, int):
    attempts_made = 0
    while attempts_made < max_attempts:
        p_init = np.random.normal(0, 1)
        h_candidate, p_candidate = hmc_sampling.integrate_trajectory(
            h_init, p_init, trajectory_length, n_steps, y_series,
            phi_input, mu_input, var_eta_input
        )
        hamiltonian_delta = hmc_sampling.hamiltonian(
            h_candidate, p_candidate, y_series, phi_input, mu_input, var_eta_input
        ) - hmc_sampling.hamiltonian(
            h_init, p_init, y_series, phi_input, mu_input, var_eta_input
        )
        if np.random.uniform() <= min(math.exp(-hamiltonian_delta), 1):
            return h_candidate, p_candidate, attempts_made
        attempts_made += 1
    return np.array([]), np.array([]), attempts_made


if __name__ == "__main__":
    pass
