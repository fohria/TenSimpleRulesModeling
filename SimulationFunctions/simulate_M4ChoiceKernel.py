import numpy as np
from numba import njit, int32

from .choose import choose


@njit
def simulate_M4ChoiceKernel(trial_count, bandit, alpha_c, beta_c):
    """
    simulate a participant using the choice kernel strategy.
    """

    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count, dtype=int32)

    CK = np.array([0.0, 0.0])  # CK = choice kernel

    for trial in range(trial_count):

        # compute choice probabilities
        probabilities = np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK))

        # make choice based on probabilities
        actions[trial] = choose(np.array([0, 1]), probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[actions[trial]] += alpha_c * 1

    return actions, rewards
