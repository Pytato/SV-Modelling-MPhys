# import numpy as np
#
# @njit
# def _h_half_step(
#         h_set_old: np.ndarray,
#         p_set_old: np.ndarray,
#         dt: float
# ) -> np.ndarray:
#     return np.add(h_set_old, (dt / 2.) * p_set_old)
#
#
# @njit
# def _p_full_step(
#         h_set_old: np.ndarray,
#         p_set_old: np.ndarray,
#         dt: float,
#         y_set: np.ndarray,
#         phi: float,
#         mu: float,
#         eta_var: float
# ) -> np.ndarray:
#     return np.add(
#         p_set_old,
#         dt * dham_by_dh_i(h_set_old, y_set, phi, mu, eta_var)
#     )
#
#
# @njit
# def _h_full_step(
#         h_set_old: np.ndarray,
#         p_set_new: np.ndarray,
#         dt: float
# ) -> np.ndarray:
#     return np.add(h_set_old, (dt/2.)*p_set_new)
#
#
# @njit
# def integrate_trajectory(
#         h_initial: np.ndarray,
#         p_initial: np.ndarray,
#         integration_length: float,
#         n_steps: int,
#         y_set: np.ndarray,
#         phi_initial: float,
#         mu_initial: float,
#         var_eta_initial: float
# ) -> [np.ndarray, np.ndarray]:
#
#     dt = integration_length/n_steps
#
#     # Initial half step update to h
#     h_set = _h_half_step(h_initial, p_initial, dt)
#
#     # Initial full-step of p to line everything up
#     p_set = _p_full_step(h_set, p_initial, dt, y_set, phi_initial, mu_initial, var_eta_initial)
#
#     for i in range(n_steps-1):
#         h_set = _h_full_step(h_set, p_set, dt)
#         p_set = _p_full_step(h_set, p_set, dt, y_set, phi_initial, mu_initial, var_eta_initial)
#
#     # Final half step to update h
#     h_set = _h_half_step(h_set, p_set, dt)
#
#     return [h_set, p_set]