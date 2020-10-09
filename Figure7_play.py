# okay, so here in figure7 we have two models: blind and state-based.
# if i understand the paper text correctly, we simulate the task with both models, and use parameters so their behaviour - in the form of their learning curves - look the same (figA)
# then we fit the state-based (not the blind) to both simulations and see that the likelihood of fit for state-based model ON blind is higher than fit for state-based model ON state-based model (figB)
# but then in figC we see that if we run simulations with the found/fitted parameters for the state-based model and the blind models, we get different behaviours, and the _behaviour_ of the state-based model fits better with figA than the _behaviour_ of the *fitted* blind model to figA.
# easy to get confused here because the colors mean different things in the 3 different plots, but in other words:
# figA: simulations for blind and state-based, lets call them blindSims and stateSims, so this is 'pure' data like plotting behaviour of participants
# figB: likelihood curves for *fitting state-based model* to blindSims and stateSims, so lets call this fitState, i.e. it's one model applied to the two datasets blindSims and stateSims
# figC: learning curves for simulations using fitState and the new fitBlind, so here it's 'pure' data again but now we use the parameter values we found for the two models

# the task as explained in the paper:
# we have three stimuli, s1, s2, s3
# three actions: a1, a2, a3
# a1 is correct choice for s1 and s2
# a3 is correct choice for s3
# a2 does nothing

# the task as inferred from the matlab code:
# we have ten blocks, called 's', in each block Q values are reset
# each block has 45 trials
# they call states 'o' assumingly for observation
# observations are in structured order; s1, s2, s3 (s1 etc for stimuli as above)

# we will use s0, s1, s2 and a0, a1, a2 instead of 1-3, so:
# three actions: a0, a1, a2
# a0 is correct choice for s0 and s1
# a2 is correct choice for s2
# a1 does nothing

# %%

# autoreload
%load_ext autoreload
%autoreload 2

import numpy as np
from numba import njit, int32, float64

import pandas as pd
import seaborn as sns

from SimulationFunctions.choose import choose

# %%
@njit
def simulate_blind(stimuli):

    alpha_pos = 0.3 + 0.4 * np.random.rand()  # paper says this is set but okay
    beta = 4 + 5 * np.random.rand()
    alpha_neg = 0

    # stimuli = np.tile(np.array([0, 1, 2], dtype=int32), int(45 / 3))  # 45 trials, 3 stimuli
    # stimuli = np.tile(np.array([0, 1, 2]), int(45 / 3))  # 45 trials, 3 stimuli

    states = np.zeros(10 * 45, dtype=int32)  # 10 blocks each with 45 trials
    actions = np.zeros(10 * 45, dtype=int32)
    rewards = np.zeros(10 * 45, dtype=int32)
    # states = np.zeros(10 * 45)  # 10 blocks each with 45 trials
    # actions = np.zeros(10 * 45)
    # rewards = np.zeros(10 * 45)

    correct_actions = np.array([0, 0, 2], dtype=int32)  # index represents state/stimuli
    # correct_actions = np.array([0, 0, 2])  # index represents state/stimuli

    for block in range(10):

        Q = np.ones(3) / 3  # each action has 1/3 change to start with

        for trial in range(45):

            observation = stimuli[trial]
            probchoice = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
            # action = np.random.choice(np.array([0, 1, 2]), p=probchoice)
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

# %%
@njit
def simulate_sights(stimuli):

    alpha_pos = 0.6 + 0.1 * np.random.rand()
    beta = 2
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

# %%


# %%
# :::: MAIN LOOP FIG A::::
# okay so next here is to be able to do many simulatinos for blind and then plot average learning curve below

sim_count = 100

# fig a data
sim_blinds = []
sim_sights = []
# np.tile is not supported by numba and is the same every loop anyway
stimuli = np.tile(np.array([0, 1, 2], dtype=int), int(45 / 3))
# fig b data
like_blinds = []
like_sights = []

for simulation in range(sim_count):

    states, actions, rewards = simulate_blind(stimuli)
    sim_blinds.append(summarize_per_stimuli(rewards))

    # stim1 = [rewards[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)]
    # stim2 = [rewards[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)]
    # stim3 = [rewards[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)]
    #
    # stim1 = np.array(stim1)
    # stim2 = np.array(stim2)
    # stim3 = np.array(stim3)
    #
    # sim_blinds.append(np.mean((stim1 + stim2 + stim3) / 3, axis=0))

    states, actions, rewards = simulate_sights(stimuli)
    sim_sights.append(summarize_per_stimuli(rewards))

    # stim1 = np.array([rewards[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)])
    # stim2 = np.array([rewards[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)])
    # stim3 = np.array([rewards[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)])
    #
    # sim_sights.append(np.mean((stim1 + stim2 + stim3) / 3, axis=0))

# %%

def summarize_per_stimuli(series):

    stim1 = np.array([series[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)])
    stim2 = np.array([series[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)])
    stim3 = np.array([series[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)])

    return np.mean((stim1 + stim2 + stim3) / 3, axis=0)

# %%

# construct likelihood function and run on latest states, actions, rewards
# then we can put it inside the loop above so likelihood is calculated for each individual simulation as well
# TODO NEXT figure out their posterior function thingie and what is exactly plotted in figB
# ::::::::: FIG B :::::::::::::::
# okay fitrl stuff works now get the L = L (1:3 etcetc)
like_blinds = []
like_sights = []
for bla in range(10):

    states, actions, rewards = simulate_blind(stimuli)

    parms, likes = fitRL(states, actions, rewards)
    like_blinds.append(summarize_per_stimuli(likes))

    # llhs1 = np.array([_likes[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)])
    # llhs2 = np.array([_likes[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)])
    # llhs3 = np.array([_likes[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)])

    # like_blinds.append(np.mean((llhs1 + llhs2 + llhs3) / 3, axis=0))

    states, actions, rewards = simulate_sights(stimuli)

    parms, likes = fitRL(states, actions, rewards)
    like_sights.append(summarize_per_stimuli(likes))

    # llhs1 = np.array([_likes[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)])
    # llhs2 = np.array([_likes[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)])
    # llhs3 = np.array([_likes[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)])
    #
    # like_sights.append(np.mean((llhs1 + llhs2 + llhs3) / 3, axis=0))

# %%
# ::::::::::::::::::: FIG B ::::::::::::::::

columns3 = ['simcount', 'trial', 'data_model', 'p_like']

df = pd.DataFrame(columns=columns3)
for simnum in range(10):
    rows = [(simnum, x, 'blind', like_blinds[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns3, data=rows)
    df = pd.concat([df, temp_df])

    rows = [(simnum, x, 'sights', like_sights[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns3, data=rows)
    df = pd.concat([df, temp_df])

fig = sns.lineplot(data=df, x='trial', y='p_like', hue='data_model')
fig.set(ylim = (0, 1))

# %%

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

# %%
from scipy.optimize import minimize

def fitRL(states, actions, rewards):
    guess = (np.random.rand(), np.random.rand() * 10)
    best = 9999
    best_parms = []
    for _ in range(20):
        result = minimize(llh_sights, (0.3, 3), args=(states, actions, rewards), bounds=[(0.01, 0.99), (0.1, 20)])
        if result.fun < best:
            best_parms = result.x

    likes = posterior(best_parms, states, actions, rewards)

    return best_parms, likes

# %%

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

# %%

# llh_sights([0.2, 7], states, actions, rewards)

# %%
# nicer way to plot with tidy data
# :::::::::::::::: FIG A:::::::::::::::::
columns2 = ['sim', 'trial', 'model', 'p_correct']

df = pd.DataFrame(columns=columns2)
for simnum in range(10):  # TODO! change to simcount
    rows = [(simnum, x, 'blind', sim_blinds[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns2, data=rows)
    df = pd.concat([df, temp_df])

    rows = [(simnum, x, 'sights', sim_sights[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns2, data=rows)
    df = pd.concat([df, temp_df])

fig = sns.lineplot(data=df, x='trial', y='p_correct', hue='model')
fig.set(ylim = (0, 1))


# %%

# NOTES DOWN HERE!

# working but ugly way to plot
columns = [f'sim{i}' for i in range(sim_count)]

df_blinds = pd.DataFrame(columns=columns, data=np.array(sim_blinds).T)
df_blinds['trial'] = np.arange(15, dtype=int)

df_sights = pd.DataFrame(columns=columns, data=np.array(sim_sights).T)
df_sights['trial'] = np.arange(15, dtype=int)

sns.set()
fig = sns.lineplot(data=df_blinds.melt('trial'), x='trial', y='value')
fig = sns.lineplot(data=df_sights.melt('trial'), x='trial', y='value')
fig.set(ylim = (0, 1));

###

test = np.mean((stim1 + stim2 + stim3) / 3, axis=0)
test2 = np.mean((stim1 + stim2 + stim3) / 2, axis=0)

np.array([test,test2]).shape
test.shape
test2.shape


np.arange(15, dtype=np.int32)
bla = pd.DataFrame(columns=['sim1', 'sim2', 'trial'], data=np.array([test,test2, np.arange(15)]).T)

bla.trial = bla.trial.astype('int')

bla.melt('trial')

sns.lineplot(data=bla.melt('trial'), x='trial', y='value')


fig = sns.lineplot(x=np.arange(15), y = np.mean((stim1 + stim2 + stim3) / 3, axis=0))
fig.set(ylim=(0,1));



np.mean(np.array([x[0] for x in (stim1 + stim2 + stim3) / 3]) / 10)

(stim1 + stim2 + stim3) / 3


rewards[np.arange(0, 45, 3)]  # stimuli 0
rewards[np.arange(1, 45, 3)]  # stimuli 1
rewards[np.arange(2, 45, 3)]  # stimuli 2

(rewards[np.arange(0, 45, 3)] + rewards[np.arange(1, 45, 3)] + rewards[np.arange(2, 45, 3)]) / 3
