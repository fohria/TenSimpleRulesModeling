import numpy as np
from numba import njit, int32

from .choose import choose


@njit
def simulate_M1random_v1(trial_count, bandit, bias):
    """
    simulates a participant making random choices with bias for one choice.

    parameters
    ----------
    trial_count : int
        number of trials, i.e. how many times do we pull a bandit arm
    bandit : numpy array of length 2
        represents a multi armed bandit. its length is the number of arms
        (NOTE: can only handle 2 arms for now) and each array item represents
        the probability of reward for pulling that bandit arm.
    bias : float (0-1)
        the bias towards the leftmost (first array position) choice

    returns
    -------
    actions : numpy array with length trial_count
        sequence of actions taken (0 or 1 for 2 arm bandit) on each trial
    rewards : numpy array with length trial_count
        sequence of rewards received on each trial
    """

    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count, dtype=int32)

    # probabilities don't change between trials
    probabilities = np.array([bias, 1 - bias])

    for trial in range(trial_count):

        # make choice according to choice probabilities
        actions[trial] = choose(np.array([0, 1]), probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

    return actions, rewards
