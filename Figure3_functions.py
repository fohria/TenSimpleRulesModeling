import numpy as np
import pandas as pd

def RLWM(rho, setsize, K, beta, alpha):

    w = rho * np.min([1, K / setsize])
    Q = 0.5 + np.zeros([setsize, 3])
    WM = 0.5 + np.zeros([setsize, 3])

    trials = np.tile(np.arange(setsize), 15)

    choices = []
    rewards = []

    for state in trials:

        # RL policy
        softmax1_value = np.exp(beta * Q[state])
        softmax1 = softmax1_value / np.sum(softmax1_value)

        # WM policy
        softmax2_value = np.exp(50 * WM[state])
        softmax2 = softmax2_value / np.sum(softmax2_value)

        # mixture policy
        probabilities = (1-w) * softmax1 + w * softmax2

        choice = np.random.choice([0, 1, 2], p=probabilities)
        choices.append(choice)

        reward = choice == (state % 3)
        rewards.append(reward)

        Q[state, choice] += alpha * (reward - Q[state, choice])
        WM[state, choice] = reward

    return trials, choices, rewards

def simulate(realrho, realK, realbeta, realalpha):
    """
        should probably have a docstring here later
    """
    column_names = ["reward", "choice", "stimulus", "setsize", "block", "trial"]
    df = pd.DataFrame(columns=column_names)

    block = -1  # to match 0-indexing and keep track of block

    for rep in range(3):
        for ns in range(2,7):  # up to but not including 7
            block += 1
            stimuli, choices, rewards = RLWM(realrho, ns, realK, realbeta, realalpha)
            assert len(stimuli) == len(choices) and len(choices) == len(rewards)
            data = np.array([
                rewards, choices, stimuli,
                np.repeat(ns, len(stimuli)),
                np.repeat(block, len(stimuli)),
                np.arange(len(stimuli))
            ])
            # such horrible naming here haha
            # TODO: put these two lines outside of loop, they take some extra time being here
            df2 = pd.DataFrame(columns=column_names, data=data.transpose())
            df = df.append(df2, ignore_index=True)

    return df


def likelihood_RLWM(rho, K, beta, alpha, blockdata):

    # to access based on trial index starting at 0 we create arrays from df
    trials  = np.array(blockdata["stimulus"])
    rewards = np.array(blockdata["reward"])
    choices = np.array(blockdata["choice"])
    setsize = np.array(blockdata["setsize"])[0]  # only need 1

    w = rho * np.min([1, K / setsize])
    Q = 0.5 + np.zeros([setsize, 3])
    WM = 0.5 + np.zeros([setsize, 3])

    loglikelihood = 0

    for trial, state in enumerate(trials):

        # RL policy
        softmax1_value = np.exp(beta * Q[state])
        # try:
        #     softmax1_value = np.exp(beta * Q[state])
        # except RuntimeWarning as error:
        #     print(f"beta is {beta} and q[state] is: {Q[state]}")
        #     print(error)
        #     raise
        softmax1 = softmax1_value / np.sum(softmax1_value)

        # WM policy
        softmax2_value = np.exp(50 * WM[state])
        # try:
        #     softmax2_value = np.exp(50 * WM[state])
        # except RuntimeWarning as error:
        #     print(f"WM[state] is {WM[state]}")
        #     print(error)
        #     raise
        softmax2 = softmax2_value / np.sum(softmax2_value)

        # mixture policy
        probabilities = (1-w) * softmax1 + w * softmax2

        # so now, based on probabilities calculated what's the probability of the choice that was made?
        prob_choice = probabilities[choices[trial]]
        loglikelihood += np.log(prob_choice)

        choice = choices[trial]
        reward = rewards[trial]
        Q[state, choice] += alpha * (reward - Q[state, choice])
        WM[state, choice] = reward

    return loglikelihood
