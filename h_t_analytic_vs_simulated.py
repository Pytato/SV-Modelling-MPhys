import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from source.analytic_funcs import analytic_funcs
from source.model_code import takaishi09_basic_sv_gen_model

# plt.tight_layout()
plt.style.use(["science", "ieee"])

TRIAL_VALUES = {
    "mu": -1,
    "phi": 0.97,
    "eta_variance": 0.05,
    "initial_volatility": 1.0,
    "h_1": 0,
    "max_t": 100000,  # 300,
}

# rc('font', **{
#     'family': 'serif',
#     'serif': ['Computer Modern'],
#     'size': '10',
# })
# rc('text', usetex=True)
# rc('figure', **{'autolayout': True})


t_set = np.arange(1, TRIAL_VALUES["max_t"]+1)

analytic_h_t_variance = analytic_funcs.variance_h_t(t_set, TRIAL_VALUES["eta_variance"], TRIAL_VALUES["phi"])
analytic_h_t_expec_value = analytic_funcs.expectation_val_h_t(t_set, TRIAL_VALUES["h_1"], TRIAL_VALUES["mu"],
                                                              TRIAL_VALUES["phi"])


_, h_t_data = takaishi09_basic_sv_gen_model.gen_y_t_vals(
    TRIAL_VALUES["phi"], TRIAL_VALUES["mu"], TRIAL_VALUES["eta_variance"], num_steps_to_gen=TRIAL_VALUES["max_t"]
)

g_mean_h_t = []
g_var_h_t = []
for i in tqdm.tqdm(range(len(h_t_data))):
    cu_h_t_data = h_t_data[:i+1]
    g_mean_h_t.append(np.mean(cu_h_t_data))
    g_var_h_t.append(np.var(cu_h_t_data))


log_err_exp = np.log10(np.abs(np.subtract(analytic_h_t_expec_value, g_mean_h_t)))
log_err_var = np.log10(np.abs(np.subtract(analytic_h_t_variance, g_var_h_t)))


fig, (ax_exp, ax_var) = plt.subplots(nrows=2, sharex="all")
ax_exp.plot(t_set, log_err_exp, linewidth=0.65)
# ax_exp.plot(t_set, g_mean_h_t, label="Rolling Measured")
ax_exp.set(ylabel=r"$\log_{10}|\Delta\langle h_t \rangle|$")
# ax_exp.legend()
ax_var.plot(t_set, log_err_var, linewidth=0.65)
# ax_var.plot(t_set, g_var_h_t, label="Measured")
ax_var.set(ylabel=r"$\log_{10}|\Delta\sigma_{h_t}^2|$", xlabel="$t$")
# ax_var.legend()
fig.savefig("./plotout/moving_theory_vs_sim_log_err.pdf")


fig, (ax_exp, ax_var) = plt.subplots(nrows=2, sharex="all")
ax_exp.plot(t_set, analytic_h_t_expec_value, label="Analytic", linewidth=0.65)
ax_exp.plot(t_set, g_mean_h_t, label="Rolling Measured", linewidth=0.65)
ax_exp.set(ylabel=r"$\langle h_t \rangle$")
ax_exp.legend()
ax_var.plot(t_set, analytic_h_t_variance, label="Analytic", linewidth=0.65)
ax_var.plot(t_set, g_var_h_t, label="Measured", linewidth=0.65)
ax_var.set(ylabel=r"$\sigma_{h_t}^2$", xlabel="$t$")
# ax_var.legend()
fig.savefig("./plotout/moving_theory_vs_sim.pdf")


# num_parallel_runs = 5000000
# h_t_data = [takaishi09_basic_sv_gen_model.gen_h_t_set(
#     TRIAL_VALUES["h_1"], TRIAL_VALUES["mu"], TRIAL_VALUES["phi"],
#     takaishi09_basic_sv_gen_model.gen_eta_t_values(TRIAL_VALUES["eta_variance"], n_to_gen=TRIAL_VALUES["max_t"]))
#         for i in tqdm.tqdm(range(num_parallel_runs))]
#
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
# # fig.savefig("./plotout/theory_vs_sim_plot.png", dpi=300)
#
# fig, (ax_expec, ax_vari) = plt.subplots(nrows=2, sharex="all")
# ax_expec.plot(t_set, analytic_h_t_expec_value, label="Analytic")
# ax_expec.plot(t_set, measured_h_t_expec_value, label="Simulated")
# ax_expec.set(ylabel=r"$\langle h_t \rangle$")
# ax_vari.plot(t_set, analytic_h_t_variance, label="Analytic")
# ax_vari.plot(t_set, measured_h_t_variance, label="Simulated")
# ax_vari.set(ylabel=r"$\sigma_{h_t}^2$", xlabel="$t$")
# # ax_vari.legend()
# ax_expec.legend()
# # fig.suptitle("Elementwise Expectation Value and Variance Plots of $h_t$ \nfor Parameters: $\\mu=-1,\\, "
# #              "\\varphi=0.97,\\, \\sigma^2_{\\eta}=0.05$.", y=1.02)
# fig.savefig("./plotout/theory_sim_plot_2.pdf")

# fig, ax = plt.subplots()
# ax.hist()

