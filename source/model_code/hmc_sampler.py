from source.model_code import leapfrog_2nd_order_integrator
from source.model_code import system_equations
# from source.model_code import mcmc_param_sampling

import numpy as np
from numba import njit


@njit
def get_new_h_vector(
        initial_h: np.ndarray,
        y_t_data: np.ndarray,
        phi: float,
        mu: float,
        var_eta: float,
        n_steps: int,
        max_attempts=50
) -> (np.ndarray, float, float, float, float):
    trajectory_length = 1.
    step_length = trajectory_length / n_steps
    attempts = 0

    while attempts < max_attempts:
        initial_p = np.random.normal(0, 1, len(initial_h))
        # phi, mu, var_eta = mcmc_param_sampling.iter_samp_param(
        #     initial_h,
        #     phi, mu, var_eta
        # )
        initial_hamiltonian = system_equations.hamiltonian(
            initial_h,
            initial_p,
            y_t_data,
            phi, mu, var_eta
        )
        candidate_h, candidate_p = leapfrog_2nd_order_integrator.full_trajectory_int(
            initial_h,
            initial_p,
            step_length,
            n_steps,
            y_t_data,
            phi, mu, var_eta
        )
        candidate_hamiltonian = system_equations.hamiltonian(
            candidate_h,
            candidate_p,
            y_t_data,
            phi, mu, var_eta
        )
        attempts += 1
        accept_prob = min(1., np.exp(initial_hamiltonian - candidate_hamiltonian))
        if np.random.uniform(0.0, 1.0) < accept_prob:
            break

    return candidate_h, attempts
