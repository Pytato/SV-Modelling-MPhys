from source.utility_code import autocorrelation

import numpy as np

from numba import njit, prange
from typing import Tuple, Callable


@njit(parallel=True)
def single_elimination_jackknife(
        sample_vect: np.ndarray,
        estimator: Callable,
        *estimator_args
) -> Tuple[float, float]:
    """Simple single elimination Jackknife estimator.

    :param sample_vect: Sample vector to apply estimator by
        jackknife on.
    :param estimator: Estimator function.
    :return: 2-tuple Estimate of the mean and standard error of given estimator
        over given sample vector, respectively.
    """

    index_set = np.arange(len(sample_vect))
    jk_permuted_estimator_results = np.array([estimator(sample_vect[index_set != i], *estimator_args)
                                              for i in prange(len(sample_vect))])
    jk_mean = np.mean(jk_permuted_estimator_results)
    jk_standard_error = np.sqrt(
        ((len(sample_vect) - 1) / len(sample_vect)) *
        np.sum(np.square(
            np.subtract(jk_permuted_estimator_results, jk_mean)
        ))
    )
    return float(jk_mean), float(np.sqrt(jk_standard_error))


@njit(parallel=True)
def n_length_bootstrap(
        sample_vect: np.ndarray,
        bootstrap_length: int,
        bootstrap_sample_count: int,
        estimator: Callable[..., float],
        *estimator_args
) -> Tuple[float, float]:
    """Simple bootstrap estimator error estimator

    :param sample_vect: Sample vector
    :param bootstrap_length: Length of bootstrap samples
    :param bootstrap_sample_count: Number of bootstrap samples to take
    :param estimator: Estimator function to evaluate
    :param estimator_args: Extra arguments for estimator function
    :return: 2-tuple Estimate of the mean and standard error of given estimator
        over given sample vector, respectively.
    """
    bootstrap_sample_set = np.random.choice(sample_vect, (bootstrap_sample_count, bootstrap_length))
    estimator_results = np.array([estimator(bootstrap_sample_set[bs_sample_index], *estimator_args)
                                  for bs_sample_index in prange(bootstrap_sample_set.shape[0])])
    estimator_bs_mean = np.sum(estimator_results)/bootstrap_sample_count
    estimator_st_err = np.sqrt(
        np.sum(
            np.square(
                np.subtract(estimator_results, estimator_bs_mean)
            )
        ) / (bootstrap_sample_count - 1)
    )

    return float(estimator_bs_mean), float(estimator_st_err)
