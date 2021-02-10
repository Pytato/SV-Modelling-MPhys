"""
Sources:
[1] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, ‘Hybrid Monte Carlo’,
    Physics Letters B, vol. 195, no. 2, pp. 216–222, Sep. 1987,
    doi: 10.1016/0370-2693(87)91197-X.
"""

from source.model_code import system_equations

import numpy as np
from numba import njit


@njit
def __p_step(
        p_set_old: np.ndarray,
        x_set_old: np.ndarray,
        dt: float,
        y_data: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> np.ndarray:
    """Constitutes an integrator step in p-space, call with half
    the usual step length to produce half-steps.
    """
    return np.subtract(
        p_set_old,
        dt*system_equations.dham_by_dh_i(
            x_set_old,
            y_data,
            phi,
            mu,
            eta_var
        ))


@njit
def __x_step(
        p_set_new: np.ndarray,
        x_set_old: np.ndarray,
        dt: float
) -> np.ndarray:
    """Constitutes an integrator step in x-space, call with half
    the usual step length to produce half-steps.
    """
    return np.add(x_set_old, dt*p_set_new)


@njit
def full_trajectory_int(
        x_initial: np.ndarray,
        p_initial: np.ndarray,
        dt: float,
        n_steps: int,
        y_data: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> [np.ndarray, np.ndarray]:
    """Integrates a full trajectory by 2nd order leapfrog, preserves
    volume and is reversible.

    ref. [1]
    """

    # Initial half step in p to align the integrator correctly
    p_curr = __p_step(p_initial, x_initial, dt/2, y_data, phi, mu, eta_var)
    # Initial full step in x to account for the loop running n-1
    x_curr = __x_step(p_curr, x_initial, dt)

    # Loop for n-1 full steps
    for i in range(n_steps-1):
        p_curr = __p_step(p_curr, x_curr, dt, y_data, phi, mu, eta_var)
        x_curr = __x_step(p_curr, x_curr, dt)

    # Final half step in p to re-align p-space and validate Hamiltonian
    p_curr = __p_step(p_curr, x_curr, dt/2, y_data, phi, mu, eta_var)

    return [x_curr, p_curr]
