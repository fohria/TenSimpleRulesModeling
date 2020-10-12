import numpy as np
from numba import njit, int32

from .choose import choose


@njit
def simulate_M5RWCK_v1(trial_count, bandit, alpha, beta, alpha_c, beta_c):
    """
    simulate a participant using a choice strategy that combines Rescorla-Wagner and choice kernel.
    """

    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count, dtype=int32)

    Q = np.array([0.5, 0.5])
    CK = np.array([0.0, 0.0])

    for trial in range(trial_count):

        # compute choice probabilities
        V = beta * Q + beta_c * CK  # V = values? vector?
        p = np.exp(V) / np.sum(np.exp(V))

        # make choice according to choice probabilities
        actions[trial] = choose(np.array([0, 1]), p)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

        # update Q values
        delta = rewards[trial] - Q[actions[trial]]
        Q[actions[trial]] += alpha * delta

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[actions[trial]] += alpha_c * 1

    return actions, rewards
