from source.model_code.takaishi09_basic_sv_gen_model import gen_y_t_vals
from source.model_code.hmc_sampling import integrate_trajectory, hamiltonian
from source.model_code.mcmc_param_sampling import iter_samp_param
from source.model_code import mcmc_fullstep_handler

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sci_stats

from tqdm import tqdm

# from matplotlib import rc

plt.style.use(["science", "ieee"])

# rc('font', **{
#     'family': 'serif',
#     'serif': ['Computer Modern'],
#     'size': '11',
# })
# rc('text', usetex=True)
# rc('figure', **{'autolayout': True})


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
    step_length_list = np.linspace(0.0001, 0.1, 5000)
    n_step_list = np.around(trajectory_length/step_length_list)
    new_hams = []
    for n_steps in tqdm(n_step_list):
        cand_h_loc, cand_p_loc = integrate_trajectory(h_loc, p_loc, trajectory_length, n_steps,
                                                      y_t_data_loc, phi_init, mu_init, eta_var_init)
        new_hams.append(hamiltonian(cand_h_loc, cand_p_loc, y_t_data_loc, phi_init, mu_init,
                                    eta_var_init))

    ham_delta_list = np.subtract(new_hams, old_ham)
    eps_square_list = np.square(np.divide(trajectory_length, n_step_list))
    rel_grad, rel_intercept, r_val, _, std_err = sci_stats.linregress(eps_square_list,
                                                                      ham_delta_list)
    fig, ax = plt.subplots()
    # ax.plot(eps_square_list, ham_delta_list, color="0.3")
    ax.plot(eps_square_list, [eps_sq*rel_grad + rel_intercept for eps_sq in eps_square_list],
            label=f"Least Squares Fitted Slope, \n"
                  f"$a={rel_grad:.2f}\\pm {std_err:.1g}$, $b={rel_intercept:.1g}$, \n"
                  f"$r={r_val:.2f}$")
    ax.set(xlabel=r"$\varepsilon^2$", ylabel=r"$\Delta H$")
    ax.legend()
    fig.suptitle("HMC Stepsize-Error Relation")
    fig.savefig("./plotout/stepsize_ham_delta.pdf")


def full_implementation_test(y_t_series, phi_init, mu_init, var_eta_init, n_trajectories):
    h_zero = np.random.normal(0, 1, len(y_t_series))
    phi_curr, mu_curr, var_eta_curr = phi_init, mu_init, var_eta_init
    h_out = h_zero
    phi_set, mu_set, var_eta_set = [phi_curr], [mu_curr], [var_eta_curr]
    h_set_set = [h_zero]
    attempts_made_on_trajec = []
    max_attempts = 200
    for _ in tqdm(range(n_trajectories)):
        h_out, (phi_curr, mu_curr, var_eta_curr), attempts_made = mcmc_fullstep_handler.step_mcmc(
            h_out, y_t_series, phi_curr, mu_curr, var_eta_curr, max_attempts=max_attempts,
            n_steps=60
        )
        attempts_made_on_trajec.append(attempts_made)
        if attempts_made == max_attempts:
            break
        phi_set.append(phi_curr)
        mu_set.append(mu_curr)
        var_eta_set.append(var_eta_curr)
        h_set_set.append(h_out)

    print("phi", phi_curr, "\nmu", mu_curr, "\nvar[eta]", var_eta_curr)
    print("Average trajectory attempts:", np.mean(attempts_made_on_trajec))
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(h_out)+1), h_out, linewidth=0.25)
    ax1.set(xlabel=r"$t$", ylabel=f"$h^{{{len(h_set_set)-1}}}_t$")
    fig1.suptitle(f"HMC Inferred $h^{{{len(h_set_set)-1}}}_t$")

    failed_traject_fig, failed_traject_ax = plt.subplots()
    failed_traject_ax.plot(np.arange(1, len(attempts_made_on_trajec)+1),
                           np.subtract(attempts_made_on_trajec, 1), linewidth=0.25)
    failed_traject_ax.set(xlabel=r"$\tau$", ylabel=f"Failed Trajectories")
    failed_traject_fig.suptitle(f"HMC Failed Trajectories Over MCMC History")

    phi_history_fig, phi_history_ax = plt.subplots()
    phi_history_ax.plot(np.arange(1, len(phi_set)+1), phi_set)
    phi_history_ax.set(xlabel=r"$\tau$", ylabel=f"\\varphi")
    phi_history_fig.suptitle(r"\varphi MC History")

    mu_history_fig, mu_history_ax = plt.subplots()
    mu_history_ax.plot(np.arange(1, len(mu_set)+1), mu_set)
    mu_history_ax.set(xlabel=r"$\tau$", ylabel=f"\\mu")
    mu_history_fig.suptitle(r"\mu MC History")

    var_eta_history_fig, var_eta_history_ax = plt.subplots()
    var_eta_history_ax.plot(np.arange(1, len(mu_set)+1), var_eta_set)
    var_eta_history_ax.set(xlabel=r"$\tau$", ylabel=f"\\sigma_\\eta^2")
    var_eta_history_fig.suptitle(r"\sigma_\eta^2")

    fig1.savefig("./plotout/hmc_full_test.pdf")
    failed_traject_fig.savefig("./plotout/hmc_failed_trajects.pdf")
    phi_history_fig.savefig("./plotout/phi_history.pdf")
    mu_history_fig.savefig("./plotout/mu_history.pdf")
    var_eta_history_fig.savefig("./plotout/var_eta_history.pdf")


if __name__ == "__main__":
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = generate_test_y_t_data(eta_var, mu, phi)
    # integrator_reverse_test(y_t_data[100000:101000])
    # sample_params(y_t_data[100000:102000], 0.5, 0.0, 1.0, 100)
    # stepsize_delta_ham_test(y_t_data[100000:101000])
    full_implementation_test(y_t_data[150000:152000], 0.5, 0, 1, 50000)
