import numpy as np
from numba import njit


@njit
def llh_sights(parameters, states, actions, rewards):

    alpha_pos = parameters[0]
    beta = parameters[1]
    alpha_neg = 0

    trialcount = len(actions)
    choice_probs = np.zeros(trialcount)

    for trial in range(trialcount):

        if trial % 45 == 0:
            Q = np.ones((3, 3)) / 3.0

        observation = states[trial]
        Q_row = Q[observation]
        prob_choice = np.exp(beta * Q_row) / np.sum(np.exp(beta * Q_row))

        action = actions[trial]
        choice_probs[trial] = prob_choice[action]

        reward = rewards[trial]
        delta = reward - Q[observation, action]
        if delta > 0:
            Q[observation, action] += alpha_pos * delta
        elif delta < 0:
            Q[observation, action] += alpha_neg * delta

    loglikelihood = np.sum(np.log(choice_probs))
    return -loglikelihood


def posterior(parms, states, actions, rewards):

    alpha_pos = parms[0]
    beta = parms[1]
    alpha_neg = 0

    trialcount = len(actions)
    likehoods = np.zeros(trialcount)

    for trial in range(trialcount):

        if trial % 45 == 0:
            Q = np.ones((3, 3)) / 3.0

        obs = states[trial]
        Q_row = Q[obs]
        probs = np.exp(beta * Q_row) / np.sum(np.exp(beta * Q_row))
        action = actions[trial]
        reward = rewards[trial]
        delta = reward - Q[obs, action]
        if delta > 0:
            Q[obs, action] += alpha_pos * delta
        elif delta < 0:
            Q[obs, action] += alpha_neg * delta

        likehoods[trial] = probs[action]

    return likehoods
