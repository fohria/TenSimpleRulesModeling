import numpy as np
from numba import njit

# https://github.com/numba/numba/issues/2539#issuecomment-507306369
# def rand_choice_nb(arr, prob):
@njit
def choose(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

# def choose(p):
#     return choice([0, 1], p=p)
