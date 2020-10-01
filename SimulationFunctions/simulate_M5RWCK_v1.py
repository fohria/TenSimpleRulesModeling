import numpy as np
from numba import njit, int32
from .choose import choose


@njit
def simulate_M5RWCK_v1(T, mu, alpha, beta, alpha_c, beta_c):

    Q = np.array([0.5, 0.5])
    CK = np.array([0.0, 0.0])

    a = np.zeros(T, dtype=int32)
    r = np.zeros(T, dtype=int32)

    for t in range(T):

        # compute choice probabilities
        V = beta * Q + beta_c * CK  # V = values? vector?
        p = np.exp(V) / np.sum(np.exp(V))

        # make choice according to choice probabilities
        a[t] = choose(np.array([0, 1]), p)

        # generate reward based on choice
        r[t] = np.random.rand() < mu[a[t]]

        # update Q values
        delta = r[t] - Q[a[t]]
        Q[a[t]] += alpha * delta

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[a[t]] += alpha_c * 1

    return a, r
