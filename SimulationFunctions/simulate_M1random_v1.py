import numpy as np
from numba import njit, int32

# from SimulationFunctions.choose import choose  # temp when working in top dir
from .choose import choose  # when using as submodule from top functions


@njit
def simulate_M1random_v1(trial_count, bandit, bias):

    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count, dtype=int32)

    probabilities = np.array([bias, 1 - bias])  # b never changes

    for trial in range(trial_count):

        # make choice according to choice probabilities
        actions[trial] = choose(np.array([0, 1]), probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

    return actions, rewards
