from source.model_code import hmc_sampler
from source.model_code import mcmc_param_sampling
from source.utility_code.autocorrelation import acf, get_autocorr_time
from source.utility_code.statistical_analysis import single_elimination_jackknife, n_length_bootstrap
from source.utility_code.testing_utils import y_t_data_management

import numpy as np
import pickle
import os

from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use(["science", "ieee"])


def acf_on_hmc_sampler_test(
        y_data: np.ndarray,
        phi_init: float,
        mu_init: float,
        eta_var_init: float,
        n_samples: int,
        hmc_int_step_count: int
):
    PERSISTENT_DATA_PATH = "./persistent_mcmc_history.pickle"
    if not os.path.exists(PERSISTENT_DATA_PATH):
        h_vect = np.random.normal(0., 1., len(y_data))
        phi_loc, mu_loc, eta_var_loc = phi_init, mu_init, eta_var_init
        attempts_list = []
        history = {"phi": [], "mu": [], "eta_var": [], "h_10": [],
                   "h_20": [], "h_100": []}
        for _ in tqdm(range(n_samples)):
            h_vect, attempts = hmc_sampler.get_new_h_vector(
                h_vect,
                y_data,
                phi_loc, mu_loc, eta_var_loc,
                hmc_int_step_count
            )
            attempts_list.append(attempts)
            phi_loc, mu_loc, eta_var_loc = mcmc_param_sampling.iter_samp_param(
                h_vect,
                phi_loc, mu_loc, eta_var_loc
            )
            history["phi"].append(phi_loc)
            history["mu"].append(mu_loc)
            history["eta_var"].append(eta_var_loc)
            history["h_10"].append(h_vect[9])
            history["h_20"].append(h_vect[19])
            history["h_100"].append(h_vect[99])
        with open(PERSISTENT_DATA_PATH, "wb") as pickle_fp:
            pickle.dump(history, pickle_fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(PERSISTENT_DATA_PATH, "rb") as pickle_fp:
            history = pickle.load(pickle_fp)

    phi_history = np.array(history["phi"][50000:])
    mu_history = history["mu"][50000:]
    eta_var_history = history["eta_var"][50000:]
    h_10 = np.array(history["h_10"][50000:150000])
    h_20 = np.array(history["h_20"][50000:150000])
    h_100 = np.array(history["h_100"][50000:150000])
    h_10_jk_mean_set = []
    h_10_jk_st_err_set = []
    h_20_jk_mean_set = []
    h_20_jk_st_err_set = []
    h_100_jk_mean_set = []
    h_100_jk_st_err_set = []
    BOOTSTRAP_SAMPLES = 30000
    BOOTSTRAP_LENGTH = 95000
    # for t in tqdm(np.arange(0, 80, step=4)):
    #     h_10_mean, h_10_st_err = n_length_bootstrap(h_10, BOOTSTRAP_SAMPLES, BOOTSTRAP_LENGTH, acf, t)
    #     h_10_jk_mean_set.append(h_10_mean)
    #     h_10_jk_st_err_set.append(h_10_st_err)
    #     h_20_mean, h_20_st_err = n_length_bootstrap(h_20, BOOTSTRAP_SAMPLES, BOOTSTRAP_LENGTH, acf, t)
    #     h_20_jk_mean_set.append(h_20_mean)
    #     h_20_jk_st_err_set.append(h_20_st_err)
    #     h_100_mean, h_100_st_err = n_length_bootstrap(h_100, BOOTSTRAP_SAMPLES, BOOTSTRAP_LENGTH, acf, t)
    #     h_100_jk_mean_set.append(h_100_mean)
    #     h_100_jk_st_err_set.append(h_100_st_err)

    time_domain = np.arange(0, 80, step=4)
    h_10_acf_test = [acf(h_10, t) for t in tqdm(time_domain)]
    h_20_acf_test = [acf(h_20, t) for t in tqdm(time_domain)]
    h_100_acf_test = [acf(h_100, t) for t in tqdm(time_domain)]

    fig, ax = plt.subplots()
    capsize = 2.5
    linewidth = 1.
    elinewidth = 0.8
    fillstyle = "none"
    # ax.errorbar(time_domain, h_10_jk_mean_set, yerr=h_10_jk_st_err_set, label="$h_{10}$", fmt="k-o",
    #             linewidth=linewidth, elinewidth=elinewidth, capsize=capsize, fillstyle=fillstyle)
    # ax.errorbar(time_domain, h_20_jk_mean_set, yerr=h_20_jk_st_err_set, label="$h_{20}$", fmt="g--D",
    #             linewidth=linewidth, elinewidth=elinewidth, capsize=capsize, fillstyle=fillstyle)
    # ax.errorbar(time_domain, h_100_jk_mean_set, yerr=h_100_jk_st_err_set, label="$h_{100}$", fmt="r:s",
    #             linewidth=linewidth, elinewidth=elinewidth, capsize=capsize, fillstyle=fillstyle)
    ax.plot(time_domain, h_100_acf_test, label="$h_{100}$")
    ax.plot(time_domain, h_20_acf_test, label="$h_{20}$")
    ax.plot(time_domain, h_10_acf_test, label="$h_{10}$")
    ax.legend(loc="upper right")
    ax.set(xlabel="Autocorrelation Time Offset", ylabel="ACF")
    fig.savefig("./plotout/no_err_h_10_20_100_acf_test.pdf")
    # ax.set_yscale("symlog", linthresh=5e-3)
    ax.set_yscale("log")
    fig.savefig("./plotout/no_err_h_10_20_100_acf_test_log_y.pdf")
    fig, ax = plt.subplots()
    phi_domain = np.arange(0, 1500, 10)
    ax.plot(phi_domain, np.abs([acf(phi_history, t) for t in phi_domain]), label="$\\varphi$")
    ax.set(xlabel="Autocorrelation Time Offset", ylabel="ACF")
    ax.set_yscale("symlog", linthresh=5e-3)
    ax.legend()
    fig.savefig("./plotout/no_err_phi_acf_test_log_y.pdf")
    print(f"phi ac time: {get_autocorr_time(phi_history)}")
    print(f"h100 ac time: {get_autocorr_time(h_100)}")


if __name__ == '__main__':
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = y_t_data_management.generate_test_y_t_data(phi, mu, eta_var)
    acf_on_hmc_sampler_test(y_t_data[120000:122000], 0.5, 0., 1., 250000, 25)
