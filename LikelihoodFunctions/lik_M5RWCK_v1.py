import numpy as np

def lik_M5RWCK_v1(parameters, actions, rewards):

    alpha = parameters[0]
    beta = parameters[1]
    alpha_c = parameters[2]
    beta_c = parameters[3]

    Q = np.array([0.5, 0.5], dtype='float128')
    CK = np.array([0, 0], dtype='float128')

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
