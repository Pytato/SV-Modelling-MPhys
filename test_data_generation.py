import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from source.model_code import takaishi09_basic_sv_gen_model

plt.tight_layout()

TRIAL_VALUES = {
    "mu": -1,
    "phi": 0.97,
    "eta_variance": 0.05,
    "initial_volatility": 1.0,
    "h_1": 0,
    "max_t": 10000,
}

rc('font', **{
    'family': 'serif',
    'serif': ['Computer Modern'],
    'size': '10',
})
rc('text', usetex=True)
rc('figure', **{'autolayout': True})

t_set = np.arange(1, TRIAL_VALUES["max_t"]+1)


fig, ax = plt.subplots()
fig.suptitle("Plot of $y_t$ Data: $\\mu=-1,\\, \\varphi=0.97,\\, \\sigma^2_{\\eta}=0.05,\\, h_1=0$.")
ax.plot(t_set, takaishi09_basic_sv_gen_model.gen_y_t_vals(
    TRIAL_VALUES["initial_volatility"], TRIAL_VALUES["eta_variance"], TRIAL_VALUES["mu"],
    TRIAL_VALUES["phi"], num_steps_to_gen=len(t_set)-1), color="0.3", linewidth=0.25)
ax.set(xlabel=r"Time, $t$", ylabel=r"$y_t$")
fig.savefig("./plotout/y-t_data_gen_test_plot.pdf")
fig.savefig("./plotout/y-t_data_gen_test_plot.png", dpi=300)

fig, ax = plt.subplots()
fig.suptitle("Plot of $h_t$ Data: $\\mu=-1,\\, \\varphi=0.97,\\, \\sigma^2_{\\eta}=0.05,\\, h_1=0$.")
ax.plot(t_set, takaishi09_basic_sv_gen_model.gen_h_t_set(
    TRIAL_VALUES["h_1"], TRIAL_VALUES["mu"], TRIAL_VALUES["phi"],
    takaishi09_basic_sv_gen_model.gen_eta_t_values(TRIAL_VALUES["eta_variance"], n_to_gen=len(t_set) - 1)),
        color="0.3", linewidth=0.4)
ax.set(xlabel=r"Time, $t$", ylabel=r"$h_t$")
fig.savefig("./plotout/h-t_data_gen_test_plot.pdf")
fig.savefig("./plotout/h-t_data_gen_test_plot.png", dpi=300)

fig, ax = plt.subplots()
fig.suptitle("Plot of $y_t$ Data: $\\mu=-1,\\, \\varphi=0.97,\\, \\sigma^2_{\\eta}=0.3,\\, h_1=0$.")
ax.plot(t_set, takaishi09_basic_sv_gen_model.gen_y_t_vals(
    TRIAL_VALUES["initial_volatility"], 0.3, TRIAL_VALUES["mu"],
    TRIAL_VALUES["phi"], num_steps_to_gen=len(t_set)-1), color="0.3", linewidth=0.25)
ax.set(xlabel=r"Time, $t$", ylabel=r"$y_t$")
fig.savefig("./plotout/y-high_eta_var_t_data_gen_test_plot.pdf")
fig.savefig("./plotout/y-high_eta_var_t_data_gen_test_plot.png", dpi=300)
