import numpy as np

from typing import Union
from numba import njit, prange
from numba.typed import List


@njit
def acf(
        var_vect: np.ndarray,
        time: int
) -> float:
    # var_vect = List(var_vect)
    exp_val = np.mean(var_vect)
    variance = np.var(var_vect)
    if time == 0:
        slice_annoyingness = var_vect
    else:
        slice_annoyingness = var_vect[:-time]
    retval = (
        np.sum(
            np.multiply(
                np.subtract(slice_annoyingness, exp_val),
                np.subtract(var_vect[time:], exp_val)
            )
        ) / ((len(var_vect)-time)*variance)
    )
    return retval


@njit(parallel=True)
def get_autocorr_time(
        var_vect: np.ndarray
) -> float:
    return np.add(
        1/2., np.sum(np.array([acf(var_vect, time) for time in prange(1, len(var_vect))]))
    )
