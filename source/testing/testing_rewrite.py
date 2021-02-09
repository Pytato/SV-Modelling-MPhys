from source.model_code import leapfrog_2nd_order_integrator
from source.model_code import hmc_sampling
from source.model_code import ideal_test_code
from source.testing import param_sampling_test

import numpy as np
from copy import deepcopy


def reversibility_test(y_data_loc):
    TRAJ_LENGTH = 1
    N_STEPS = 50
    dt = TRAJ_LENGTH/N_STEPS
    PHI_INIT, MU_INIT, ETA_VAR_INIT = 0.5, 0.0, 1.0
    x_set_in = np.ones_like(y_data_loc)/2.
    p_set_in = np.random.normal(0, 2, len(y_data_loc))
    x_traject_end_f, p_traject_end_f = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_set_in,
        p_set_in,
        hmc_sampling.dham_by_dh_i,
        dt,
        N_STEPS,
        y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT
    )
    x_traject_end_a, p_traject_end_a = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_set_in,
        p_set_in,
        ideal_test_code.force,
        dt,
        N_STEPS,
        y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT
    )

    print(np.abs(np.subtract(x_traject_end_a, x_traject_end_f)))
    print(np.abs(np.subtract(p_traject_end_a, p_traject_end_f)))

    # Now the sign of p will be flipped, setting the integrator to retrace the path it took
    # before.
    p_traject_end_a = -p_traject_end_a
    p_traject_end_f = -p_traject_end_f

    x_traject_start_f, p_traject_start_f = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_traject_end_f,
        p_traject_end_f,
        hmc_sampling.dham_by_dh_i,
        dt,
        N_STEPS,
        y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT
    )
    x_traject_start_a, p_traject_start_a = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_traject_end_a,
        p_traject_end_a,
        ideal_test_code.force,
        dt,
        N_STEPS,
        y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT
    )

    print(np.abs(np.subtract(x_traject_start_f, x_set_in)))
    print(np.abs(np.subtract(p_traject_start_f, p_set_in)))
    print(np.abs(np.subtract(x_traject_start_a, x_set_in)))
    print(np.abs(np.subtract(p_traject_start_a, p_set_in)))


if __name__ == "__main__":
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = param_sampling_test.generate_test_y_t_data(eta_var, mu, phi)
    reversibility_test(y_t_data)
