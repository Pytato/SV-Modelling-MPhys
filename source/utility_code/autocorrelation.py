import numpy as np

from typing import Union
from numba import njit


@njit
def acf(
        var_vect: Union[np.ndarray, list],
        time: int,
        exp_val: float,
        variance: float
) -> np.ndarray:
    if time == 0:
        slice_annoyingness = var_vect
    else:
        slice_annoyingness = var_vect[:-time]
    return (
        np.sum(
            np.multiply(
                np.subtract(slice_annoyingness, exp_val),
                np.subtract(var_vect[time:], exp_val)
            )
        ) / ((len(var_vect)-time)*variance)
    )
