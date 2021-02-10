from source.model_code.takaishi09_basic_sv_gen_model import gen_y_t_vals

import os
import numpy as np


def generate_test_y_t_data(eta_var_l, mu_l, phi_l):
    DATA_FILE_LOC = "./y_t_model_data_output.npy"
    if os.path.exists(DATA_FILE_LOC):
        temp = np.load(DATA_FILE_LOC)
        return temp

    model_output_data = gen_y_t_vals(eta_var_l, mu_l, phi_l, num_steps_to_gen=300000)
    np.save(DATA_FILE_LOC, model_output_data)
    return model_output_data
