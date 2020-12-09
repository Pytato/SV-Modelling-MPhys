from source.code.model_code.takaishi09_basic_sv_gen_model import gen_y_t_vals
from source.code.model_code.hmc_sampling import integrate_trajectory

import os
import numpy as np


def generate_test_y_t_data(*args):
    DATA_FILE_LOC = "./y_t_model_data_output.npy"
    if os.path.exists(DATA_FILE_LOC):
        return np.load(DATA_FILE_LOC)

    model_output_data = gen_y_t_vals(*args, num_steps_to_gen=300000)
    np.save(DATA_FILE_LOC, model_output_data)


def sample_params(y_t_data_internal, phi_init, mu_init, var_eta_init):
    h_t_gen_set, (phi_val, mu_val, var_eta_val) = integrate_trajectory(
        np.ones_like(y_t_data_internal), np.zeros_like(y_t_data_internal),
        1, 100000, y_t_data_internal, phi_init, mu_init, var_eta_init
    )
    print(phi_val, mu_val, var_eta_val)


if __name__ == "__main__":
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = generate_test_y_t_data(eta_var, mu, phi)
    sample_params(y_t_data[100000:105000], 0.5, 0.0, 0.1)
