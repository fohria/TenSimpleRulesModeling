import numpy as np

def lik_M2WSLS_v1(parameters, actions, rewards):

    epsilon = parameters[0]

    # last reward/action (initialize as None)
    last_reward = None
    last_action = None

    trialcount = len(actions)
    choice_probabilities = np.zeros(trialcount)

    # loop over all trials
    for trial in range(trialcount):

        # compute choice probabilities
        if last_reward is None:

            # choose randomly on first trial
            p = [0.5, 0.5]

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
