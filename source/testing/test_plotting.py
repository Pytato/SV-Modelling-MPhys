from source.model_code import mcmc_param_sampling as param_sampling
from source.model_code import takaishi09_basic_sv_gen_model as sv_gen_model
from source.testing import mcmc_param_sampler_arb_dists as arb_dist_model

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats

import csv

from tqdm import tqdm

# from matplotlib import rc
#
#
# rc('font', **{
#     'family': 'serif',
#     'serif': ['Computer Modern'],
#     'size': '10',
# })
# rc('text', usetex=True)
# rc('figure', **{'autolayout': True})
plt.style.use(["science", "ieee"])


def test_dists_arb():
    A_VAL = 20.0
    B_VAL = 2.0
    C_VAL = 5.0
    D_VAL = 2
    E_VAL = 1.5
    N_VAL = 100
    VAR_ETA_VAL_MU = 0.5
    VAR_ETA_VAL_PHI = 0.1

    eta_var_set = [0.5]
    mu_set = [0.1]
    phi_set = [0.3]

    sample_count = 10000000

    for i in tqdm(range(sample_count)):
        phi_set.append(arb_dist_model.phi_sample(phi_set[-1], D_VAL, E_VAL, VAR_ETA_VAL_PHI))
    make_standard_hist(phi_set[10000:], "phi_sample_dist_plot_arb_new",
                       title=r"$\varphi$ Sample Distribution", n_bins=100,
                       x_lab=r"$\varphi$", y_lab="Normalised Frequency",
                       over_plot_data=[[
                           np.linspace(min(phi_set), max(phi_set), num=400),
                           phi_analytic_func(np.linspace(min(phi_set), max(phi_set), num=400), D_VAL,
                                             E_VAL, VAR_ETA_VAL_PHI),
                           "Analytic Phi Distribution"
                       ]])

    for i in tqdm(range(sample_count)):
        mu_set.append(arb_dist_model.mu_sample(B_VAL, C_VAL, VAR_ETA_VAL_MU))
    make_standard_hist(mu_set[10000:], "mu_sample_dist_plot_arb", title=r"$\mu$ Sample Distribution",
                       n_bins=100,
                       x_lab=r"$\mu$", y_lab="Normalised Frequency",
                       over_plot_data=[standard_gaussian(mu_set[10000:])])

    for i in tqdm(range(4000000)):
        eta_var_set.append(arb_dist_model.eta_var_sample(A_VAL, N_VAL))
    make_standard_hist(eta_var_set[10000:], "eta_var_sample_dist_plot_arb",
                       title=r"$\sigma^2_\eta$ Sample Distribution", n_bins=100,
                       x_lab=r"$\sigma^2_\eta$",
                       y_lab="Normalised Frequency",
                       over_plot_data=[inv_gamma_fit(eta_var_set[10000:])])


def test_eta_var_inv_gamma():
    INITIAL_H = 0.0
    ETA_T_VAR = 0.05
    MU = -1.0
    PHI = 0.97
    eta_t_set = sv_gen_model.gen_eta_t_values(ETA_T_VAR, n_to_gen=210000)
    h_t_test_data = sv_gen_model.gen_h_t_set(INITIAL_H, MU, PHI, eta_t_set)
    # deal with burn-in
    historical_eta_t = eta_t_set[50000:]
    historical_h_t = h_t_test_data[50000:51000]

    phi_set = [0.0]
    eta_var_set = [0.0]
    mu_set = [0.0]

    for i in tqdm(range(len(historical_h_t))):
        new_phi, new_eta_var, new_mu = param_sampling.iter_samp_param(historical_h_t, phi_set[-1],
                                                                      mu_set[-1], eta_var_set[-1])
        phi_set.append(new_phi)
        eta_var_set.append(new_eta_var)
        mu_set.append(new_mu)

    print(f"""
phi: {phi_set[-1]:.4},
eta_var: {eta_var_set[-1]:.4},
mu: {mu_set[-1]:.4}.
""")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(phi_set) + 1), phi_set)
    ax.set_ylim(0.90, 1)
    fig.savefig("./plotout/test_phi_sample_plot.png", dpi=300)
    # print(param_sampling.rejections)
    # print(param_sampling.total_samples)
    # print(param_sampling.rejections/param_sampling.total_samples)

    fig, ax = plt.subplots()
    ax.hist(phi_set)
    fig.show()


def make_standard_hist(
        distributed_variable_set,
        plot_name_no_ext,
        n_bins=80,
        title="",
        x_lab="",
        y_lab="",
        over_plot_data=None
):
    fig, ax = plt.subplots()
    bin_freq, bins, _ = ax.hist(distributed_variable_set, n_bins, density=True, color="0.3",
                                align="mid")
    if over_plot_data is not None:
        for overplot_domain, overplot_vals, name in over_plot_data:
            ax.plot(overplot_domain, overplot_vals, linewidth=0.5)  # , label=name)
        ax.legend()
    ax.set(xlabel=x_lab, ylabel=y_lab)
    # fig.suptitle(title)
    fig.savefig(f"./plotout/{plot_name_no_ext}.pdf")
    fig.savefig(f"./plotout/{plot_name_no_ext}.png", dpi=300)


def standard_gaussian(
        distributed_variable_set
):
    data_set_mean = np.mean(distributed_variable_set)
    data_set_var = np.var(distributed_variable_set)
    min_domain_val = min(distributed_variable_set)
    max_domain_val = max(distributed_variable_set)
    domain = np.linspace(min_domain_val, max_domain_val, num=1000)
    return [domain, np.exp(-np.square(domain - data_set_mean) / (2 * data_set_var)) / np.sqrt(
        2 * np.pi * data_set_var),
            f"Fitted Gaussian: \n$\\mu={data_set_mean:.4}$, $\\sigma^2={data_set_var:.2}$"]


def phi_analytic_func(
        phi: np.ndarray,
        d_val: float,
        e_val: float,
        eta_var: float
) -> np.ndarray:
    pre_norm_ret_set = np.multiply(
        np.sqrt(1 - np.square(phi)),
        np.exp(-(d_val/(2*eta_var))*np.square(phi - (e_val/d_val)))
    )
    norm_const = np.sum(pre_norm_ret_set * (np.ptp(phi)/len(phi)))
    return pre_norm_ret_set/norm_const


def inv_gamma_fit(
        distributed_variable_set
):
    mle_tuple = scistats.invgamma.fit(distributed_variable_set)
    min_domain_val = min(distributed_variable_set)
    max_domain_val = max(distributed_variable_set)
    domain = np.linspace(min_domain_val, max_domain_val, num=1000)
    return [domain, scistats.invgamma.pdf(domain, *mle_tuple),
            f'Fitted Inverse Gamma']  # : \n$\\mu={-1*mle_tuple[1]:.3}$, $\\sigma^2={mle_tuple[2]**(-1):.2}$']


def dummy_h_dist_tests():
    INITIAL_H = 0.0
    ETA_T_VAR = 0.05
    MU = -1.0
    PHI = 0.97
    # eta_t_set = sv_gen_model.gen_eta_t_values(ETA_T_VAR, n_to_gen=100000)
    # h_t_test_data = sv_gen_model.gen_h_t_set(INITIAL_H, MU, PHI, eta_t_set)[80000:90000]
    h_t_test_data = np.random.uniform(size=5000) * 4.0 - 1.0
    # h_t_test_data = np.genfromtxt("./Single_run.csv", delimiter=",")[80000:90000]
    phi_set = [0.8]
    eta_var_set = [0.08]
    mu_set = [0.4]
    for i in tqdm(range(2000000)):
        new_phi, new_eta_var, new_mu = param_sampling.iter_samp_param(h_t_test_data, phi_set[-1],
                                                                      mu_set[-1], eta_var_set[-1])
        phi_set.append(new_phi)
        eta_var_set.append(new_eta_var)
        mu_set.append(new_mu)

    with open("./newrun_out.csv", "w") as csv_out:
        writer = csv.writer(csv_out, delimiter=",")
        writer.writerows(
            [[phi_set[i], eta_var_set[i], mu_set[i]] for i in tqdm(range(len(phi_set)))])

    # print(param_sampling.phi_rejects)
    # print(param_sampling.phi_samples)
    # print(param_sampling.phi_rejects/param_sampling.phi_samples)

    make_standard_hist(phi_set[100:], "phi_sample_dist_plot3",
                       title=r"$\varphi$ Sample Distribution", n_bins=100,
                       x_lab=r"$\varphi$", y_lab="Count",
                       over_plot_data=[standard_gaussian(phi_set[100:])])

    make_standard_hist(eta_var_set[100:], "eta_var_sample_dist_plot3",
                       title=r"$\sigma^2_\eta$ Sample Distribution", n_bins=100,
                       x_lab=r"$\sigma^2_\eta$",
                       y_lab="Count", over_plot_data=[inv_gamma_fit(eta_var_set[100:]),
                                                      standard_gaussian(eta_var_set[100:])])

    make_standard_hist(mu_set[100:], "mu_sample_dist_plot3", title=r"$\mu$ Sample Distribution",
                       n_bins=100,
                       x_lab=r"$\mu$", y_lab="Count",
                       over_plot_data=[standard_gaussian(mu_set[100:])])


if __name__ == "__main__":
    # test_eta_var_inv_gamma()
    # dummy_h_dist_tests()
    test_dists_arb()
