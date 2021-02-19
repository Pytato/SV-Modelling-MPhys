from source.model_code import hmc_sampler
from source.model_code import mcmc_param_sampling
from source.utility_code.autocorrelation import acf
from source.utility_code.testing_utils import y_t_data_management

import numpy as np

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

    # phi_history, mu_history, eta_var_history = history["phi"], history["mu"], history["eta_var"]
    h_10 = history["h_10"][10000:]
    h_20 = history["h_20"][10000:]
    h_100 = history["h_100"][10000:]

    h_100_expval = float(np.mean(h_100))
    h_100_variance = float(np.var(h_100))
    h_100_acf_test = [acf(h_100, t, h_100_expval, h_100_variance) for t in tqdm(np.arange(0, 80))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 80), np.abs(h_100_acf_test))
    ax.set(xlabel="Autocorrelation Time Offset", ylabel="ACF", yscale="log")
    fig.savefig("./plotout/h_100_acf_test.pdf")

    h_20_expval = float(np.mean(h_20))
    h_20_variance = float(np.var(h_20))
    h_20_acf_test = [acf(h_20, t, h_20_expval, h_20_variance) for t in tqdm(np.arange(0, 80))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 80), np.abs(h_20_acf_test))
    ax.set(xlabel="Autocorrelation Time Offset", ylabel="ACF", yscale="log")
    fig.savefig("./plotout/h_20_acf_test.pdf")

    h_10_expval = float(np.mean(h_20))
    h_10_variance = float(np.var(h_20))
    h_10_acf_test = [acf(h_10, t, h_10_expval, h_10_variance) for t in tqdm(np.arange(0, 80))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 80), np.abs(h_10_acf_test))
    ax.set(xlabel="Autocorrelation Time Offset", ylabel="ACF", yscale="log")
    fig.savefig("./plotout/h_10_acf_test.pdf")


if __name__ == '__main__':
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = y_t_data_management.generate_test_y_t_data(phi, mu, eta_var)
    acf_on_hmc_sampler_test(y_t_data[120000:122000], 0.5, 0., 1., 250000, 25)
