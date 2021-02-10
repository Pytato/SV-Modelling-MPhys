from numba import njit
import numpy as np


@njit
def hamiltonian(
        h_set: np.ndarray,
        p_set: np.ndarray,
        y_set: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> float:

    ham_first_ele = np.sum(np.square(p_set))
    ham_second_ele = np.sum(np.add(h_set, np.multiply(np.square(y_set), np.exp(-1. * h_set))))
    ham_third_ele = ((h_set[0] - mu)**2)/(eta_var/(1. - phi*phi))
    ham_fourth_ele = (1./eta_var)*np.sum(
        np.square(
            np.add(h_set[1:] - mu, -phi*(h_set[:-1] - mu))
        )
    )
    return 0.5 * (ham_first_ele + ham_second_ele + ham_third_ele + ham_fourth_ele)


@njit
def dham_by_dh_i(
        h_set: np.ndarray,
        y_set: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> np.ndarray:

    # h_set_extended = np.pad(h_set, 1)
    h_set_extended = np.zeros(len(h_set)+2)
    h_set_extended[1:-1] = h_set
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
                    np.subtract(one_arr, first_ele_kron_handler)
                ),
                np.multiply(
                    -phi*np.add(
                        h_set_extended[2:] - mu,
                        -phi*(h_set - mu)
                    ),
                    np.subtract(one_arr, final_ele_kron_handler)
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
        h_set_old: np.ndarray,
        p_set_old: np.ndarray,
        dt: float,
        y_set: np.ndarray,
        phi: float,
        mu: float,
        eta_var: float
) -> np.ndarray:
    return np.add(
        p_set_old,
        dt * dham_by_dh_i(h_set_old, y_set, phi, mu, eta_var)
    )


@njit
def _h_full_step(
        h_set_old: np.ndarray,
        p_set_old: np.ndarray,
        dt: float
) -> np.ndarray:
    return np.add(
        h_set_old, (dt/2.)*p_set_old
    )


@njit
def integrate_trajectory(
        h_initial: np.ndarray,
        p_initial: np.ndarray,
        integration_length: float,
        n_steps: int,
        y_set: np.ndarray,
        phi_initial: float,
        mu_initial: float,
        var_eta_initial: float
) -> [np.ndarray, np.ndarray]:

    dt = integration_length/n_steps

    # Initial half step update to h
    h_set = _h_half_step(h_initial, p_initial, dt)

    # Initial full-step of p to line everything up
    p_set = _p_full_step(h_set, p_initial, dt, y_set, phi_initial, mu_initial, var_eta_initial)

    for i in range(n_steps-1):
        h_set = _h_full_step(h_set, p_set, dt)
        p_set = _p_full_step(h_set, p_set, dt, y_set, phi_initial, mu_initial, var_eta_initial)

    # Final half step to update h
    h_set = _h_half_step(h_set, p_set, dt)

    return [h_set, p_set]
