from source.model_code import mcmc_param_sampling
from source.utility_code.testing_utils import y_t_data_management

import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use(["science", "ieee"])


def __sampler_acc_test(h_t_data):
    phi_init, mu_init, eta_var_init = 0.5, 0., 1.
    phi_samples = [phi_init]
    mu_samples = [mu_init]
    eta_var_samples = [eta_var_init]
    for i in tqdm(range(50000)):
        phi_samples.append(mcmc_param_sampling.phi_sample(
            phi_samples[i],
            h_t_data,
            mu_samples[-1],
            eta_var_samples[-1]
        ))
        mu_samples.append(mcmc_param_sampling.mu_sample(
            h_t_data,
            phi_samples[-1],
            eta_var_samples[-1]
        ))
        eta_var_samples.append(mcmc_param_sampling.var_eta_sample(
            h_t_data,
            phi_samples[-1],
            mu_samples[-1]
        ))
    fig, ax = plt.subplots()
    ax.plot(np.add(range(50000), 1)[10000:], phi_samples[10001:])
    fig.savefig("./less_saturated_plotout/phi_sampler_test.pdf")
    fig, ax = plt.subplots()
    ax.plot(np.add(range(50000), 1)[10000:], mu_samples[10001:])
    fig.savefig("./less_saturated_plotout/mu_sampler_test.pdf")
    fig, ax = plt.subplots()
    ax.plot(np.add(range(50000), 1)[10000:], eta_var_samples[10001:])
    fig.savefig("./less_saturated_plotout/eta_var_sampler_test.pdf")


if __name__ == '__main__':
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data_glob, h_t_data_glob = y_t_data_management.generate_test_y_t_data(phi, mu, eta_var)
    __sampler_acc_test(h_t_data_glob)
