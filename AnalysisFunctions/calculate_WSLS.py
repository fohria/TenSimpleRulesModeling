import numpy as np


def winstay_losestay(actions, rewards):
    """
    calculate win-stay lose-stay scores for a sequence of actions and rewards.
    """

    # compare the last action to the current action at each time step
    last_actions = np.append(-1, actions[:-1])
    stay = last_actions == actions

    last_rewards = np.append(-1, rewards[:-1])

    win_stay = np.mean(stay[last_rewards == 1])
    lose_stay = np.mean(stay[last_rewards == 0])

    return win_stay, lose_stay


def winstay_loseshift(actions, rewards):
    """
    calculate win-stay lose-shift scores for a sequence of actions and rewards.
    """

    # compare the last action to the current action at each time step
    last_actions = np.append(-1, actions[:-1])
    stay = last_actions == actions
    shift = last_actions != actions  # note that added -1 will count as shift

    last_rewards = np.append(-1, rewards[:-1])

    win_stay = np.mean(stay[last_rewards == 1])
    lose_shift = shift[last_rewards == 0]  # above -1 won't matter in the end

    return win_stay, lose_shift
