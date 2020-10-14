import numpy as np
from numba import njit, int32


@njit
def likelihood_RLWM_block(alpha, beta, rho, K, blockdata):

    stimuli = blockdata[0]
    choices = blockdata[1]
    rewards = blockdata[2]
    setsize = len(np.unique(stimuli))

    w = rho * np.min(np.array([1, K / setsize]))
    Q = 0.5 + np.zeros((setsize, 3))
    WM = 0.5 + np.zeros((setsize, 3))

    loglikelihood = 0

    for trial, state in enumerate(stimuli):

        # RL policy
        p_rl = np.exp(beta * Q[state]) / np.sum(np.exp(beta * Q[state]))

        # WM policy
        p_wm = np.exp(50 * WM[state]) / np.sum(np.exp(50 * WM[state]))

        # mixture policy
        probabilities = (1 - w) * p_rl + w * p_wm

        # so now, based on probabilities calculated what's the probability of the choice that was made?
        prob_choice = probabilities[choices[trial]]
        loglikelihood += np.log(prob_choice)

        choice = choices[trial]
        reward = rewards[trial]
        delta = reward - Q[state, choice]
        Q[state, choice] += alpha * delta
        WM[state, choice] = reward

    return loglikelihood
