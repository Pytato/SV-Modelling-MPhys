import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from source.analytic_funcs import analytic_funcs
from source.model_code import takaishi09_basic_sv_gen_model

plt.tight_layout()

TRIAL_VALUES = {
    "mu": -1,
    "phi": 0.97,
    "eta_variance": 0.05,
    "initial_volatility": 1.0,
    "h_1": 0,
    "max_t": 500,
}

rc('font', **{
    'family': 'serif',
    'serif': ['Computer Modern'],
    'size': '10',
})
rc('text', usetex=True)
rc('figure', **{'autolayout': True})


t_set = np.arange(1, TRIAL_VALUES["max_t"]+1)

analytic_h_t_variance = analytic_funcs.variance_h_t(t_set, TRIAL_VALUES["eta_variance"], TRIAL_VALUES["phi"])
analytic_h_t_expec_value = analytic_funcs.expectation_val_h_t(t_set, TRIAL_VALUES["h_1"], TRIAL_VALUES["mu"],
                                                              TRIAL_VALUES["phi"])

num_parallel_runs = 5000000
h_t_data = [takaishi09_basic_sv_gen_model.gen_h_t_set(
    TRIAL_VALUES["h_1"], TRIAL_VALUES["mu"], TRIAL_VALUES["phi"],
    takaishi09_basic_sv_gen_model.gen_eta_t_values(TRIAL_VALUES["eta_variance"], n_to_gen=TRIAL_VALUES["max_t"] - 1))
        for i in tqdm.tqdm(range(num_parallel_runs))]

# measured_h_t_variance = np.var(h_t_data, axis=0)
# measured_h_t_expec_value = np.mean(h_t_data, axis=0)
#
# log_abs_diff_expec_value = np.log10(np.abs(np.subtract(analytic_h_t_expec_value, measured_h_t_expec_value)))
# log_abs_diff_variance = np.log10(np.abs(np.subtract(analytic_h_t_variance, measured_h_t_variance)))
#
# fig, (ax_expec, ax_vari) = plt.subplots(nrows=2, sharex="all")
# fig.suptitle("Log$_{10}$ Error Plots for Parameters: $\\mu=-1,\\, \\varphi=0.97,\\, "
#              "\\sigma^2_{\\eta}=0.05,\\, h_1=0$,\n Averaged Across "+f"{num_parallel_runs:,} runs.")
# ax_expec.set(ylabel=r"$\log_{10}{|\langle h_{t_{exact}}\rangle - \langle h_{t_{gen}} \rangle|}$")
# ax_expec.plot(t_set, log_abs_diff_expec_value, color="0.3")
# y_ex_start, y_ex_end = ax_expec.get_ylim()
# ax_expec.yaxis.set_ticks(np.arange(round(y_ex_start), y_ex_end, 1))
# ax_vari.set(ylabel=r"$\log_{10}{|\mathrm{Var}[h_{t_{exact}}] - \mathrm{Var}[h_{t_{gen}}]|}$", xlabel="Time Step, $t$")
# ax_vari.plot(t_set, log_abs_diff_variance, color="0.3")
# fig.savefig("./plotout/theory_vs_sim_plot.pdf")
# fig.savefig("./plotout/theory_vs_sim_plot.png", dpi=300)

fig, ax = plt.subplots()
ax.hist()

