import numpy as np
from numba import njit, int32

from .choose import choose


@njit
def simulate_RLWM_block(alpha, beta, rho, K, setsize):

    w = rho * np.min(np.array([1, K / setsize]))
    Q = 0.5 + np.zeros((setsize, 3))
    WM = 0.5 + np.zeros((setsize, 3))

    trials = get_trial_sequence(setsize)

    choices = np.zeros(len(trials), dtype=int32)
    rewards = np.zeros(len(trials), dtype=int32)

    for trial, state in enumerate(trials):

        # RL policy using softmax
        p_rl = np.exp(beta * Q[state]) / np.sum(np.exp(beta * Q[state]))

        # WM policy using softmax
        p_wm = np.exp(50 * WM[state]) / np.sum(np.exp(50 * WM[state]))

        # mixture policy
        probabilities = (1 - w) * p_rl + w * p_wm

        choice = choose(np.array([0, 1, 2]), probabilities)
        choices[trial] = choice

        reward = choice == (state % 3)
        rewards[trial] = reward

        delta = reward - Q[state, choice]
        Q[state, choice] += alpha * delta
        WM[state, choice] = reward

    return trials, choices, rewards


@njit
def get_trial_sequence(setsize):
    """
    Create trial sequence.
    Example:
    Sequence of the form [0, 1, 2, 0, 1, 2] if setsize = 3
    Numba doesn't support numpy.tile but loves loops.
    """

    sequence = []
    for _ in range(15):
        for trial in range(setsize):
            sequence.append(trial)

    return np.array(sequence, dtype=int32)
