import numpy as np
from numba import njit
from tqdm import tqdm


@njit
def x_half_step(x_old, p_old, dt):
    return np.add(x_old, (dt/2) * p_old)


@njit
def p_step(p_old, dham_by_dx, dt, *ham_args):
    return np.subtract(p_old, dt * dham_by_dx(*ham_args))


@njit
def x_full_step(x_half_step_results, p_step_results, dt):
    return np.add(x_half_step_results, (dt/2) * p_step_results)


@njit
def integrate_trajectory_jit(x_initial, p_initial, dham_by_dx_func, path_length, n_steps, *ham_args):
    dt = path_length/n_steps
    x_set = x_initial
    p_set = p_initial

    for i in range(n_steps):
        x_half_step_set = x_half_step(x_set, p_set, dt)
        p_set = p_step(p_set, dham_by_dx_func, dt, *ham_args)
        x_set = x_full_step(x_half_step_set, p_set, dt)

    return x_set, p_set


def integrate_trajectory(x_initial, p_initial, dham_by_dx_func, path_length, n_steps, *ham_args):
    dt = path_length/n_steps
    x_set = x_initial
    p_set = p_initial

    for _ in tqdm(range(n_steps)):
        x_half_step_set = x_half_step(x_set, p_set, dt)
        p_set = p_step(p_set, dham_by_dx_func, dt, *ham_args)
        x_set = x_full_step(x_half_step_set, p_set, dt)

    return x_set, p_set
