import numpy as np
from numba import njit, int32

from .choose import choose


@njit
def simulate_M3RescorlaWagner_v1(trial_count, bandit, alpha, beta):
    """
    simulate a participant using Rescora-Wagner choice strategy.
    """

    actions = np.zeros(T, dtype=int32)
    rewards = np.zeros(T, dtype=int32)

    # the Q matrix represents what "value" each choice has to the agent
    Q = np.array([0.5, 0.5])  # init with equal probabilities for each action

    for trial in range(trial_count):

        # compute choice probabilities (softmax)
        probabilities = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

        # make choice based on choice probabilities
        actions[trial] = choose(np.array([0, 1]), probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

        # update action values
        delta = rewards[trial] - Q[actions[trial]]  # in paper this is called prediction error
        Q[actions[trial]] += alpha * delta

    return actions, rewards
