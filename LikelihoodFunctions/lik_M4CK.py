import numpy as np
from numba import njit


@njit
def lik_M4CK(parameters, actions, rewards):

    alpha_c = parameters[0]
    beta_c = parameters[1]

    CK = np.array([0.0, 0.0])

    trialcount = len(actions)
    choice_probabilities = np.zeros(trialcount)

    for trial in range(trialcount):

        # compute choice probabilities
        p = np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK))

        # compute choice probability for actual choice
        choice_probabilities[trial] = p[actions[trial]]

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[actions[trial]] += alpha_c * 1

    # compute loglikelihood and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
