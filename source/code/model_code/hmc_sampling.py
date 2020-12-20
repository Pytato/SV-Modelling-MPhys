from source.code.model_code.mcmc_param_sampling import *

from tqdm import tqdm
from numba import njit
import numpy as np


@njit
def _dham_by_dh_i(
        h_set: np.ndarray,
        y_set: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> np.ndarray:

    h_set_extended = np.pad(h_set, 1)
    first_ele_kron_handler = np.zeros_like(h_set)
    final_ele_kron_handler = np.zeros_like(h_set)
    first_ele_kron_handler[0] = 1
    final_ele_kron_handler[-1] = 1
    one_arr = np.ones_like(first_ele_kron_handler)
    dham_by_dh_i = np.add(
        np.add(
            0.5 * (1. - np.multiply(np.square(y_set), np.exp(-1. * h_set))),
            first_ele_kron_handler * ((h_set[0] - mu)/(eta_var/(1.-(phi*phi))))
        ),
        (1/eta_var) * (
            np.add(
                np.multiply(
                    np.add(
                        h_set - mu,
                        -phi*(h_set_extended[0:-2] - mu)
                    ),
                    one_arr - first_ele_kron_handler
                ),
                np.multiply(
                    -phi*np.add(
                        h_set_extended[2:] - mu,
                        -phi*(h_set - mu)
                    ),
                    one_arr - final_ele_kron_handler
                )
            )
        )
    )

    return dham_by_dh_i


@njit
def _h_half_step(
        h_set_old: np.ndarray,
        p_set_old: np.ndarray,
        dt: float
) -> np.ndarray:
    return np.add(h_set_old, (dt / 2.) * p_set_old)


@njit
def _p_full_step(
        p_set_old: np.ndarray,
        h_set_old: np.ndarray,
        dt: float,
        y_set: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> np.ndarray:
    return np.subtract(
        p_set_old,
        dt * _dham_by_dh_i(h_set_old, y_set, phi, mu, eta_var)
    )


@njit
def _integration_full_step(
        p_set_old: np.ndarray,
        h_set_old: np.ndarray,
        dt: float,
        y_set: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> (np.ndarray, np.ndarray):
    p_full_step_set = _p_full_step(p_set_old, h_set_old, dt, y_set, phi, mu, eta_var)
    return (np.add(
        _h_half_step(h_set_old, p_set_old, dt),
        (dt / 2) * p_full_step_set
    ), p_full_step_set)


def integrate_trajectory(
        p_initial: np.ndarray,
        h_initial: np.ndarray,
        integration_length: float,
        n_steps: int,
        y_set: np.ndarray,
        phi_initial: float,
        mu_initial: float,
        var_eta_initial: float
) -> (np.ndarray, (float, float, float)):

    dt = integration_length/n_steps
    h_set = h_initial
    p_set = p_initial
    phi_val = phi_initial
    mu_val = mu_initial
    var_eta_val = var_eta_initial

    for i in tqdm(range(n_steps)):
        h_set, p_set = _integration_full_step(p_set, h_set, dt, y_set, phi_val, mu_val, var_eta_val)
        phi_val, var_eta_val, mu_val = iter_samp_param(h_set, phi_val, var_eta_val, mu_val)

    return h_set, (phi_val, mu_val, var_eta_val)

