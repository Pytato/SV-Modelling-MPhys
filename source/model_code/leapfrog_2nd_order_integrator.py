"""
Sources:
[1] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, ‘Hybrid Monte Carlo’,
    Physics Letters B, vol. 195, no. 2, pp. 216–222, Sep. 1987,
    doi: 10.1016/0370-2693(87)91197-X.

"""

from typing import Callable

import numpy as np
from numba import njit


def __p_step(
        p_set_old: np.ndarray,
        x_set_old: np.ndarray,
        force_func: Callable,
        dt: float,
        *force_args
) -> np.ndarray:
    """Constitutes an integrator step in p-space, call with half
    the usual step length to produce half-steps.
    """
    return np.subtract(p_set_old, dt*force_func(x_set_old, *force_args))


def __x_step(
        p_set_new: np.ndarray,
        x_set_old: np.ndarray,
        dt: float
) -> np.ndarray:
    """Constitutes an integrator step in x-space, call with half
    the usual step length to produce half-steps.
    """
    return np.add(x_set_old, dt*p_set_new)


def full_trajectory_int(
        x_initial: np.ndarray,
        p_initial: np.ndarray,
        force_func: Callable,
        dt: float,
        n_steps: int,
        *force_args
) -> [np.ndarray, np.ndarray]:
    """Integrates a full trajectory by 2nd order leapfrog, preserves
    volume and is reversible.

    Based on [1].
    """

    # Initial half step in p to align the integrator correctly
    p_curr = __p_step(p_initial, x_initial, force_func, dt/2, *force_args)
    # Initial full step in x to account for the loop running n-1
    x_curr = __x_step(p_curr, x_initial, dt)

    # Loop for n-1 full steps
    for i in range(n_steps-1):
        p_curr = __p_step(p_curr, x_curr, force_func, dt, *force_args)
        x_curr = __x_step(p_curr, x_curr, dt)

    # Final half step in p to re-align p-space and validate Hamiltonian
    p_curr = __p_step(p_curr, x_curr, force_func, dt/2, *force_args)

    return [x_curr, p_curr]
