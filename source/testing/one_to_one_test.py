from source.model_code import ideal_test_code
from source.model_code import hmc_sampling

import numpy as np


phi_init, mu_init, eta_var_init = 0.5, 0.0, 1.0
ARB_TEST_LEN = 1500
arb_test_h = np.random.normal(0, 2, ARB_TEST_LEN)  # np.ones(ARB_TEST_LEN)/2
arb_test_y = np.random.normal(0, 2, ARB_TEST_LEN)
arb_test_p = np.random.normal(0, 2, ARB_TEST_LEN)

a_force_out = ideal_test_code.force(arb_test_h, arb_test_y, phi_init, mu_init, eta_var_init)
a_ham_out = ideal_test_code.H(arb_test_h, arb_test_p, arb_test_y, phi_init, mu_init, eta_var_init)

f_force_out = hmc_sampling.dham_by_dh_i(arb_test_h, arb_test_y, phi_init, mu_init, eta_var_init)
f_ham_out = hmc_sampling.hamiltonian(arb_test_h, arb_test_p, arb_test_y, phi_init, mu_init, eta_var_init)

force_diff = np.abs(np.subtract(a_force_out, f_force_out))
ham_diff = np.abs(np.subtract(a_ham_out, f_ham_out))

print(force_diff)
print(ham_diff)



