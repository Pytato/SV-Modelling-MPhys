from source.model_code import leapfrog_2nd_order_integrator
from source.model_code import system_equations
from source.model_code import ideal_test_code
from source.utility_code.testing_utils import y_t_data_management

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats as scistats

plt.style.use(["science", "ieee"])


def reversibility_test(y_data_loc):
    TRAJ_LENGTH = 1
    N_STEPS = 50
    dt = TRAJ_LENGTH/N_STEPS
    PHI_INIT, MU_INIT, ETA_VAR_INIT = 0.5, 0.0, 1.0
    x_set_in = np.ones_like(y_data_loc)/2.
    p_set_in = np.random.normal(0, 2, len(y_data_loc))
    x_traject_end_f, p_traject_end_f = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_set_in,
        p_set_in,
        dt,
        N_STEPS,
        y_data_loc,
        PHI_INIT,
        MU_INIT,
        ETA_VAR_INIT
    )
    x_traject_end_a, p_traject_end_a = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_set_in,
        p_set_in,
        dt,
        N_STEPS,
        y_data_loc,
        PHI_INIT,
        MU_INIT,
        ETA_VAR_INIT
    )

    print(np.abs(np.subtract(x_traject_end_a, x_traject_end_f)))
    print(np.abs(np.subtract(p_traject_end_a, p_traject_end_f)))

    # Now the sign of p will be flipped, setting the integrator to retrace the path it took
    # before.
    p_traject_end_a = -p_traject_end_a
    p_traject_end_f = -p_traject_end_f

    x_traject_start_f, p_traject_start_f = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_traject_end_f,
        p_traject_end_f,
        dt,
        N_STEPS,
        y_data_loc,
        PHI_INIT,
        MU_INIT,
        ETA_VAR_INIT
    )
    x_traject_start_a, p_traject_start_a = leapfrog_2nd_order_integrator.full_trajectory_int(
        x_traject_end_a,
        p_traject_end_a,
        ideal_test_code.force,
        dt,
        N_STEPS,
        y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT
    )

    print(np.abs(np.subtract(x_traject_start_f, x_set_in)))
    print(np.abs(np.subtract(-p_traject_start_f, p_set_in)))
    print(np.abs(np.subtract(x_traject_start_a, x_set_in)))
    print(np.abs(np.subtract(-p_traject_start_a, p_set_in)))


def area_preservation_test(y_data_loc):
    PHI_INIT, MU_INIT, ETA_VAR_INIT = 0.5, 0.0, 1.0
    x_set_in = np.ones_like(y_data_loc) / 2.
    p_set_in = np.random.normal(0, 2, len(y_data_loc))
    TRAJ_LENGTH = 1
    step_lengths = [1e-4, 2e-3, 3e-3, 4e-3, 5e-3]
    n_steps = np.around(np.divide(TRAJ_LENGTH, step_lengths), decimals=0).astype(int)
    output_tuples = []
    for i in range(len(step_lengths)):
        output_tuples.append(
            leapfrog_2nd_order_integrator.full_trajectory_int(
                x_set_in,
                p_set_in,
                step_lengths[i],
                n_steps[i],
                y_data_loc,
                PHI_INIT,
                MU_INIT,
                ETA_VAR_INIT
            )
        )
    step_squared_domain = np.square(step_lengths)
    output_hamiltonians = [
        system_equations.hamiltonian(h_set, p_set, y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT)
        for h_set, p_set in output_tuples
    ]
    hamiltonian_errors = np.abs(np.subtract(
        system_equations.hamiltonian(x_set_in, p_set_in, y_data_loc, PHI_INIT, MU_INIT, ETA_VAR_INIT),
        output_hamiltonians
    ))
    result = scistats.linregress(step_squared_domain, hamiltonian_errors)
    fig, ax = plt.subplots()
    ax.scatter(step_squared_domain, hamiltonian_errors, color="0.1", marker="x")
    ax.plot(step_squared_domain, result.slope*step_squared_domain + result.intercept,
            label=f"$a={result.slope:.1f}$, $b={result.intercept:.1g}$, $r={result.rvalue:.1g}$")
    ax.set(xlabel="$\\varepsilon^2$", ylabel="$\\Delta H$")
    ax.legend()
    fig.savefig("./less_saturated_plotout/eps_squared_plot.pdf")


if __name__ == "__main__":
    eta_var, mu, phi = 0.05, -1.0, 0.97
    y_t_data, h_t_data = y_t_data_management.generate_test_y_t_data(eta_var, mu, phi)
    # reversibility_test(y_t_data[160000:162000])
    area_preservation_test(y_t_data[160000:162000])
