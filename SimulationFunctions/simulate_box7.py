import numpy as np
from numba import njit, int32, float64
from .choose import choose


@njit
def simulate_blind(stimuli, alpha_pos, beta):

    alpha_neg = 0

    states = np.zeros(10 * 45, dtype=int32)  # 10 blocks each with 45 trials
    actions = np.zeros(10 * 45, dtype=int32)
    rewards = np.zeros(10 * 45, dtype=int32)

    # index represents state/stimuli
    correct_actions = np.array([0, 0, 2], dtype=int32)

    for block in range(10):

        Q = np.ones(3) / 3  # each action has 1/3 change to start with

        for trial in range(45):

            observation = stimuli[trial]
            probchoice = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

            action = choose(np.array([0, 1, 2], dtype=int32), probchoice)
            reward = action == correct_actions[observation]

            delta = reward - Q[action]
            if delta > 0:
                Q[action] += alpha_pos * delta
            elif delta < 0:  # paper doesnt define delta == 0
                Q[action] += alpha_neg * delta

            save_index = block * 45 + trial
            states[save_index] = observation
            actions[save_index] = action
            rewards[save_index] = reward

    return states, actions, rewards


@njit
def simulate_sights(stimuli, alpha_pos, beta):

    alpha_neg = 0

    states = np.zeros(10 * 45, dtype=int32)
    actions = np.zeros(10 * 45, dtype=int32)
    rewards = np.zeros(10 * 45, dtype=int32)

    correct_actions = np.array([0, 0, 2], dtype=int32)

    for block in range(10):

        Q = np.ones((3, 3)) / 3.0

        for trial in range(45):

            observation = stimuli[trial]
            Q_row = Q[observation]
            prob_choice = np.exp(beta * Q_row) / np.sum(np.exp(beta * Q_row))

            action = choose(np.array([0, 1, 2], dtype=int32), prob_choice)
            reward = action == correct_actions[observation]

            delta = reward - Q[observation, action]
            if delta > 0:
                Q[observation, action] += alpha_pos * delta
            elif delta < 0:
                Q[observation, action] += alpha_neg * delta

            save_index = block * 45 + trial
            states[save_index] = observation
            actions[save_index] = action
            rewards[save_index] = reward

    return states, actions, rewards
