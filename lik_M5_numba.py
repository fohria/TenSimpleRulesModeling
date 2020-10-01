import numpy as np
# from numba import jit, float64, int64, boolean
from numba import jit

# @jit(float64(float64[:], int64[:], boolean[:]), nopython=True, parallel=True)
# @jit(float64(float64[:], int64[:], boolean[:]), nopython=True)
# @jit(nopython=True, parallel=True)  # omp_set_nested routine deprecated
@jit(nopython=True)
def lik_M5_numba(parameters, actions, rewards):

    alpha = parameters[0]
    beta = parameters[1]
    alpha_c = parameters[2]
    beta_c = parameters[3]

    # Q = np.array([0.5, 0.5], dtype='np.float128')
    # CK = np.array([0, 0], dtype='np.float128')
    Q = np.array([0.5, 0.5])  # let numba decide what it is
    CK = np.array([0.0, 0.0])  # important with .0 or numba will make it int64!

    trialcount = len(actions)
    choice_probabilities = np.zeros(trialcount)

    for trial in range(trialcount):

        # compute choice probabilities
        V = beta * Q + beta_c * CK
        p = np.exp(V) / np.sum(np.exp(V))

        # compute choice probability for actual choice
        choice_probabilities[trial] = p[actions[trial]]

        # update values
        delta = rewards[trial] - Q[actions[trial]]
        Q[actions[trial]] += alpha * delta

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[actions[trial]] += alpha_c * 1

    # compute loglikelihood and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
