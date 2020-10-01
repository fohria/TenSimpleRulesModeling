import numpy as np
from numba import njit


@njit
def lik_M2WSLS_v1(parameters, actions, rewards):

    epsilon = parameters[0]

    # last reward/action (initialize as -1)
    last_reward = np.int(-1)
    last_action = np.int(-1)

    trialcount = len(actions)
    choice_probabilities = np.zeros(trialcount)

    for trial in range(trialcount):

        # compute choice probabilities
        if last_reward == -1:

            # choose randomly on first trial
            p = np.array([0.5, 0.5])

        else:

            # choice depends on last reward
            if last_reward == 1:

                # winstay (with probability 1-epsilon)
                p = epsilon / 2 * np.array([1, 1])
                p[last_action] = 1 - epsilon / 2

            else:

                # lose shift (with probability 1-epsilon)
                p = (1 - epsilon / 2) * np.array([1, 1])
                p[last_action] = epsilon / 2

        # compute choice probability for actual choice
        choice_probabilities[trial] = p[actions[trial]]

        last_action = actions[trial]
        last_reward = rewards[trial]

    # compute log-likelihood and return negative loglikelihood
    loglike = np.sum(np.log(choice_probabilities))
    return -loglike
