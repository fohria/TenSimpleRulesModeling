import numpy as np
from numba import njit, int32

# from SimulationFunctions.choose import choose  # temp when working in top dir
from .choose import choose  # when using as submodule from top functions

@njit
def simulate_M1random_v1(T, mu, b):

    a = np.zeros(T, dtype=int32)
    r = np.zeros(T, dtype=int32)

    p = np.array([b, 1 - b])  # b never changes. just like war. war never changes.

    for t in range(T):

        # compute choice probabilities
        # p = [b, 1-b]  # this could be put outside for loop

        # make choice according to choice probabilities
        # a[t] = np.random.choice([0, 1], p=p)
        a[t] = choose(np.array([0, 1]), p)  # numba doesnt like np.random.choice

        # generate reward based on choice
        r[t] = np.random.rand() < mu[a[t]]

    return a, r
