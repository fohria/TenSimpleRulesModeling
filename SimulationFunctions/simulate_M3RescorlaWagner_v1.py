import numpy as np
from numba import njit, int32
from .choose import choose


@njit
def simulate_M3RescorlaWagner_v1(T, mu, alpha, beta):

    Q = np.array([0.5, 0.5])  # init with equal values for each action
    a = np.zeros(T, dtype=int32)
    r = np.zeros(T, dtype=int32)

    for t in range(T):

        # compute choice probabilities (softmax)
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

        # make choice based on choice probabilities
        a[t] = choose(np.array([0, 1]), p)

        # generate reward based on choice
        r[t] = np.random.rand() < mu[a[t]]

        # update action values
        delta = r[t] - Q[a[t]]  # in paper this is called prediction error
        Q[a[t]] += alpha * delta

    return a, r
