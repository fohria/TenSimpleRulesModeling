import numpy as np

def lik_M3RescorlaWagner_v1(parameters, a, r):

    alpha = parameters[0]
    beta = parameters[1]

    Q = np.array([0.5, 0.5])

    trial_count = len(a)
    choice_probabilities = np.zeros(trial_count)

    # loop over all trials
    for trial in range(trial_count):

        # compute choice probabilities
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q));

        # add choice probability for actual choice
        choice_probabilities[trial] = p[a[trial]];

        # update values
        delta = r[trial] - Q[a[trial]]  # aka prediction error
        Q[a[trial]] = Q[a[trial]] + alpha * delta;

    # compute negative log-likelihood
    return -np.sum(np.log(choice_probabilities));
