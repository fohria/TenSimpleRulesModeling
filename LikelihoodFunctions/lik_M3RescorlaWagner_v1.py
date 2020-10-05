import numpy as np
from numba import njit

@njit
def lik_M3RescorlaWagner_v1(parameters, actions, rewards):

    alpha = parameters[0]
    beta = parameters[1]

    Q = np.array([0.5, 0.5])

    trial_count = len(actions)
    choice_probabilities = np.zeros(trial_count)

    for trial in range(trial_count):

        # compute choice probabilities
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[actions[trial]]

        # update values
        delta = rewards[trial] - Q[actions[trial]]  # aka prediction error
        Q[actions[trial]] += alpha * delta

    # compute negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
