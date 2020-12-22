from source.code.model_code.takaishi09_basic_sv_gen_model import gen_y_t_vals
from source.code.model_code.hmc_sampling import integrate_trajectory, hamiltonian
from source.code.model_code.mcmc_param_sampling import iter_samp_param

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as sci_opt

from tqdm import tqdm

from matplotlib import rc


rc('font', **{
    'family': 'serif',
    'serif': ['Computer Modern'],
    'size': '11',
})
rc('text', usetex=True)
rc('figure', **{'autolayout': True})


def generate_test_y_t_data(eta_var_l, mu_l, phi_l):
    DATA_FILE_LOC = "./y_t_model_data_output.npy"
    if os.path.exists(DATA_FILE_LOC):
        temp = np.load(DATA_FILE_LOC)
        return temp

    model_output_data = gen_y_t_vals(eta_var_l, mu_l, phi_l, num_steps_to_gen=300000)
    np.save(DATA_FILE_LOC, model_output_data)
    return model_output_data


def sample_params(y_t_data_loc, phi_init, mu_init, var_eta_init, n_trajectories):
    h_loc = np.ones_like(y_t_data_loc) / 2.
    phi_loc, mu_loc, var_eta_loc = phi_init, mu_init, var_eta_init
    trajectory_length = 1.
    n_steps = 10
    p_loc = np.random.normal(0.0, 1.0, len(y_t_data_loc))
    hamiltonian_arr = [hamiltonian(h_loc, p_loc, y_t_data_loc, phi_loc, mu_loc, var_eta_loc)]
    print("Params (phi, mu, eta variance):", phi_loc, mu_loc, var_eta_loc)
    rejection_count = 0
    accept_count = 0

    while len(hamiltonian_arr)-1 < n_trajectories:
        old_ham = hamiltonian(h_loc, p_loc, y_t_data_loc, phi_loc, mu_loc, var_eta_loc)
        h_cand_loc, p_cand_loc = integrate_trajectory(h_loc, p_loc, trajectory_length, n_steps,
                                                      y_t_data_loc, phi_loc, mu_loc, var_eta_loc)
        new_ham = hamiltonian(h_cand_loc, p_cand_loc, y_t_data_loc, phi_loc, mu_loc, var_eta_loc)
        ham_delta = new_ham - old_ham
        accept_prob = min([1., math.exp(-ham_delta)])
        # print(accept_prob)
        if np.random.uniform() <= accept_prob:
            accept_count += 1
            h_loc = h_cand_loc
            phi_loc, mu_loc, var_eta_loc = iter_samp_param(h_loc, phi_loc, mu_loc, var_eta_loc)
            hamiltonian_arr.append(new_ham)
            print("Params (phi, mu, eta variance):", phi_loc, mu_loc, var_eta_loc)
            print("Hamiltonian Delta:", ham_delta)
            print(f"Acceptance: {100 * (accept_count / (accept_count + rejection_count))}%")
        else:
            rejection_count += 1
        p_loc = np.random.normal(0.0, 1.0, len(y_t_data_loc))
    # print("Hamiltonian List:", hamiltonian_arr)


def integrator_reverse_test(y_t_data_loc):
    h_loc = np.ones_like(y_t_data_loc)/2
    phi_init, mu_init, eta_var_init = 0.5, 0.0, 1.0
    p_loc = np.random.normal(0, 1, len(y_t_data_loc))
    print(h_loc)
    # print("Hamiltonian initial:",
    #       hamiltonian(h_loc, p_loc, y_t_data_loc, phi_init, mu_init, eta_var_init))
    h_cand_out, p_cand_out = integrate_trajectory(h_loc, p_loc, 1., 1000, y_t_data_loc, phi_init,
                                                  mu_init, eta_var_init)
    print(h_cand_out)
    h_reverse_test, p_reverse_test = integrate_trajectory(h_cand_out, -p_cand_out, 1., 1000,
                                                          y_t_data_loc, phi_init, mu_init,
                                                          eta_var_init)
    print(h_reverse_test)


def stepsize_delta_ham_test(y_t_data_loc):
    h_loc = np.ones_like(y_t_data_loc)/2
    phi_init, mu_init, eta_var_init = 0.5, 0.0, 1.0
    p_loc = np.random.normal(0, 1, len(y_t_data_loc))
    old_ham = hamiltonian(h_loc, p_loc, y_t_data_loc, phi_init, mu_init, eta_var_init)
    trajectory_length = 1.
    # n_step_list = np.linspace(10, 1000, 1000)
    # step_length_list = np.logspace(-3, -1, 1000)
    step_length_list = np.linspace(0.0001, 0.1, 5000)
    n_step_list = np.around(trajectory_length/step_length_list)
    new_hams = []
    for n_steps in tqdm(n_step_list):
        cand_h_loc, cand_p_loc = integrate_trajectory(h_loc, p_loc, trajectory_length, n_steps,
                                                      y_t_data_loc, phi_init, mu_init, eta_var_init)
        new_hams.append(hamiltonian(cand_h_loc, cand_p_loc, y_t_data_loc, phi_init, mu_init,
                                    eta_var_init))

    ham_delta_list = np.subtract(new_hams, old_ham)
    fig, ax = plt.subplots()
    ax.plot(np.square(np.divide(trajectory_length, n_step_list)), ham_delta_list, color="0.3")
    ax.set(xlabel=r"$\varepsilon^2$", ylabel=r"$\Delta H$")
    fig.suptitle("HMC Stepsize-Error Relation")
    fig.savefig("./plotout/stepsize_ham_delta.pdf")


if __name__ == "__main__":
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = generate_test_y_t_data(eta_var, mu, phi)
    # integrator_reverse_test(y_t_data[100000:101000])
    # sample_params(y_t_data[100000:102000], 0.5, 0.0, 1.0, 100)
    stepsize_delta_ham_test(y_t_data[100000:101000])
