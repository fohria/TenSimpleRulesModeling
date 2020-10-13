import numpy as np
from numba import njit


# numba doesn't understand numpy's random choice function
# https://github.com/numba/numba/issues/2539#issuecomment-507306369
@njit
def choose(array, probabilities):
    """
    :param array: A 1D numpy array of values to sample from.
    :param probabilities: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return array[np.searchsorted(
        np.cumsum(probabilities), np.random.random(), side="right"
    )]


# def choose(p):
#     return np.random.choice([0, 1], p=p)
