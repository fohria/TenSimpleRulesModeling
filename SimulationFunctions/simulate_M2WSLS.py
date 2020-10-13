import numpy as np
from numba import njit, int32

from .choose import choose


@njit
def simulate_M2WSLS(trial_count, bandit, epsilon):
    """
    simulates a participant making choices using a "win-stay lose-shift" strategy.
    """

    # last reward/action initialized to -1 so numba knows type
    last_reward = np.int(-1)
    last_action = np.int(-1)

    # initialize the action and reward lists
    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count, dtype=int32)

    for trial in range(trial_count):

        if last_reward == -1:  # choose randomly on first trial

            probabilities = np.array([0.5, 0.5])

        else:  # choice depends on last trial

            if last_reward == 1:  # win-stay (with probability 1 - epsilon)

                probabilities = (epsilon / 2) * np.array([1, 1])
                probabilities[last_action] = 1 - epsilon / 2

            else:  # lose-shift (probability 1 - epsilon)

                probabilities = (1 - epsilon / 2) * np.array([1, 1])
                probabilities[last_action] = epsilon / 2

        # make choice according to choice probabilities
        actions[trial] = choose(np.array([0, 1]), probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

        # set last action and reward to current before next loop iteration
        last_action = actions[trial]
        last_reward = rewards[trial]

    return actions, rewards
