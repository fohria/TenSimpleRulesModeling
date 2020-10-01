import numpy as np
from numba import njit


# python's minimize function needs parameters to be a list
@njit
def lik_M1random_v1(parameters, actions, rewards):

    bias = parameters[0]
    # note rewards array is not used here but included to fit notation better with other likelihood functions

    trial_count = len(actions)
    choice_probabilities = np.zeros(trial_count)

    p = np.array([bias, 1 - bias])  # bias doesn't change

    for trial in range(trial_count):

        # compute choice probabilities
        # p = [bias, 1 - bias]

        # compute choice probability for actual choice
        choice_probabilities[trial] = p[actions[trial]]

    # compute log-likelihood and return negative loglikelihood
    loglike = np.sum(np.log(choice_probabilities))
    return -loglike
