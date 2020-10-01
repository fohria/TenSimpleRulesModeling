import numpy as np
from numba import njit, int32
from .choose import choose


@njit
def simulate_M2WSLS_v1(T, mu, epsilon):

    # last reward/action initialized to -1 so numba knows type
    rLast = np.int(-1)
    aLast = np.int(-1)

    # initialize the action and reward lists
    a = np.zeros(T, dtype=int32)
    r = np.zeros(T, dtype=int32)

    for t in range(T):

        # compute choice probabilities
        if rLast == -1:

            # choose randomly on first trial
            p = np.array([0.5, 0.5])

        else:

            # choice depends on last reward
            if rLast == 1:

                # winstay (with probability 1-epsilon)
                p = (epsilon / 2) * np.array([1, 1])
                p[aLast] = 1 - epsilon / 2

            else:

                # lose shift (probability 1-epsilon)
                p = (1 - epsilon / 2) * np.array([1, 1])
                p[aLast] = epsilon / 2

        # make choice according to choice probabilities
        a[t] = choose(np.array([0, 1]), p)

        # generate reward based on choice
        r[t] = np.random.rand() < mu[a[t]]

        # set last action and reward to current before next loop iteration
        aLast = a[t]
        rLast = r[t]

    return a, r
