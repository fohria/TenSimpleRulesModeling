# okay, so here in figure7 we have two models: blind and state-based.
# if i understand the paper text correctly, we simulate the task with both models, and use parameters so their behaviour - in the form of their learning curves - look the same (figA)
# then we fit the state-based (not the blind) to both simulations and see that the likelihood of fit for state-based model ON blind is higher than fit for state-based model ON state-based model (figB)
# but then in figC we see that if we run simulations with the found/fitted parameters for the state-based model and the blind models, we get different behaviours, and the _behaviour_ of the state-based model fits better with figA than the _behaviour_ of the *fitted* blind model to figA.
# easy to get confused here because the colors mean different things in the 3 different plots, but in other words:
# figA: simulations for blind and state-based, lets call them blindSims and stateSims, so this is 'pure' data like plotting behaviour of participants
# figB: likelihood curves for *fitting state-based model* to blindSims and stateSims, so lets call this fitState, i.e. it's one model applied to the two datasets blindSims and stateSims
# figC: learning curves for simulations using parameters from fitState to simulate state-based, so we use the parameter values we found and simulate the _state based_
# ~and the new fitBlind, so here it's 'pure' data again but now we use the parameter values we found for the two models~

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

# redescribing fig7 after reproducing all the graphs:
# figa shows how we simulate two artificial participants; blind and sighted
# figb shows how we model the two participants with the sighted/state-based model. in other words, we fit the sighted model to both blind and sighted data. interestingly, it seems the sighted model has a higher likelihood for explaining the blind data
# however, if we use the fitted parameters found in figb and simulate using the sighted model, we find that the behaviour produced (in the form of learning curves) works with the parameters fit to sighted data, but the parameters fit to blind data produces behaviour that doesn't look like the behaviour in figa.
# the last bit there in figc can be a bit confusing, but it basically says that if we fit sighted model to blind data, the parameters we then find, if we use those to simulate the sighted model, we get behaviour that doesn't look right.

# %%

# autoreload
%load_ext autoreload
%autoreload 2

import numpy as np
from numba import njit, int32, float64

import pandas as pd
import seaborn as sns

from scipy.optimize import minimize

from SimulationFunctions.choose import choose

# %%
@njit
def simulate_blind(stimuli, alpha_pos, beta):

    # alpha_pos = 0.3 + 0.4 * np.random.rand()  # paper says this is set but okay
    # beta = 4 + 5 * np.random.rand()
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
def simulate_sights(stimuli, alpha_pos, beta):

    # alpha_pos = 0.6 + 0.1 * np.random.rand()
    # beta = 2
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

testa = df_a.copy()#.head(30)
testb = df_b.copy()#.head(30)
testc = df_c.copy()#.head(30)

# df_combo = pd.DataFrame(columns=['simcount', 'trial', 'model', 'p_figa', 'p_figb', 'p_figc'])

testa = testa.rename(columns={'sim': 'simcount'})
# testa = testa.rename(columns={'p_correct': 'p_figa'})
testa['fig'] = 'a'

testb = testb.rename(columns={'data_model': 'model'})
testb = testb.rename(columns={'p_like': 'p_correct'})
testb['fig'] = 'b'

testc = testc.rename(columns={'sim_model': 'model'})
# testc = testc.rename(columns={'p_correct': 'p_figc'})
testc['fig'] = 'c'

# df_combo = testa.merge(testb).merge(testc)
df_combo = testa.append(testb).append(testc)

df_combo

# %%
# wait okay, so what we need is a column 'fig' with a, b, c and then the p_correct value and the 'easy' way to do that is just loop through but i GUESS we might be able to pivot/groupby or something?

# seems to be very difficult to set y-axes labels individually

retrowave = sns.color_palette(palette=["#f174a8",  "#7ae9b7", "#c782f5", "#47b9df", "#ade464"])
# retrowave = ["#f174a8", "#c782f5", "#47b9df", "#7ae9b7", "#ade464"]
sns.set_palette(retrowave)

sns.set(rc={'axes.facecolor':'#444444', 'figure.facecolor':'#282828', 'axes.edgecolor':'#282828', 'figure.edgecolor': '#db02ac', 'text.color': '#abb2be', 'axes.labelcolor': '#abb2be', 'xtick.color': '#abb2be', 'ytick.color': '#abb2be', 'font.family': 'Helvetica', 'axes.grid': 'True', 'grid.color': '#282828', 'axes.spines.left': 'False'})

# sns.set()

grid = sns.FacetGrid(df_combo, col='fig', hue='model')
grid.map(sns.lineplot, 'trial', 'p_correct')
grid.add_legend()
grid.set_xlabels("time step")
grid.set(ylim=(0.2, 1), xlim=(-1, 15), xticks=[0, 5, 10, 14])
grid.set_ylabels("p(correct)")
# grid.despine()
axes = grid.axes.flatten()
axes[0].set_title("'subject' learning curves")
axes[0].set_ylabel("p(correct)")
axes[1].set_title("likelihood of \nstate-based RL model")
axes[1].set_ylabel("likelihood of choice")
axes[2].set_title("simulated learning curves \nfrom state-based RL")
axes[2].set_ylabel("p(correct)")



# fig = sns.lineplot(data=df_combo, x='trial', y='p_figb', hue='model')
# fig.set(ylim=(0.2, 1));


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
# fig c data
simfit_blind = []
simfit_sight = []

for simulation in range(sim_count):

    print(f"simulation {simulation}")

    # blind model
    # alpha_pos = 0.3 + 0.4 * np.random.rand()  # paper says this is set but okay
    alpha_pos = 0.5
    beta = 6.5
    # beta = 4 + 5 * np.random.rand()
    states, actions, rewards = simulate_blind(
        stimuli, alpha_pos, beta)
    sim_blinds.append(summarize_per_stimuli(rewards))

    parms, likes = fitRL(states, actions, rewards)
    like_blinds.append(summarize_per_stimuli(likes))

    # simulation of _sighted_ model using blind data
    states, actions, rewards = simulate_sights(
        stimuli, parms[0], parms[1]
    )
    simfit_blind.append(summarize_per_stimuli(rewards))

    # state-based model
    # alpha_pos = 0.6 + 0.1 * np.random.rand()
    alpha_pos = 0.65
    beta = 2
    states, actions, rewards = simulate_sights(
        stimuli, alpha_pos, beta)
    sim_sights.append(summarize_per_stimuli(rewards))

    parms, likes = fitRL(states, actions, rewards)
    like_sights.append(summarize_per_stimuli(likes))

    states, actions, rewards = simulate_sights(
        stimuli, parms[0], parms[1])
    simfit_sight.append(summarize_per_stimuli(rewards))


# %%

def summarize_per_stimuli(series):

    stim1 = np.array([series[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)])
    stim2 = np.array([series[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)])
    stim3 = np.array([series[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)])

    return np.mean((stim1 + stim2 + stim3) / 3, axis=0)

# %%

# ::::::::::::::::::: FIG C ::::::::::::::::

columns_c = ['simcount', 'trial', 'sim_model', 'p_correct']

df_c = pd.DataFrame(columns=columns_c)
for simnum in range(sim_count):
    rows = [(simnum, x, 'blind', simfit_blind[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_c, data=rows)
    df_c = pd.concat([df_c, temp_df])

    rows = [(simnum, x, 'sights', simfit_sight[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_c, data=rows)
    df_c = pd.concat([df_c, temp_df])

sns.set(rc={"figure.figsize": (3, 3), "figure.dpi": 100})
fig = sns.lineplot(data=df_c, x='trial', y='p_correct', hue='sim_model')
# fig.set(ylim = (0, 1));
fig.set(ylim = (0.2, 1));

# %%
# ::::::::::::::::::: FIG B ::::::::::::::::

columns_b = ['simcount', 'trial', 'data_model', 'p_like']

df_b = pd.DataFrame(columns=columns_b)
for simnum in range(sim_count):
    rows = [(simnum, x, 'blind', like_blinds[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_b, data=rows)
    df_b = pd.concat([df_b, temp_df])

    rows = [(simnum, x, 'sights', like_sights[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_b, data=rows)
    df_b = pd.concat([df_b, temp_df])

sns.set(rc={"figure.figsize": (3, 3), "figure.dpi": 100})
fig = sns.lineplot(data=df_b, x='trial', y='p_like', hue='data_model')
# fig.set(ylim = (0, 1));
fig.set(ylim = (0.2, 1));

fig = sns.lineplot(data=df_combo, x='trial', y='p_figb', hue='model')
fig.set(ylim=(0.2, 1));


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


def fitRL(states, actions, rewards):

    best = 9999
    best_parms = []
    for _ in range(20):
        guess = (np.random.rand(), np.random.rand() * 10)
        result = minimize(llh_sights, guess, args=(states, actions, rewards), bounds=[(0.01, 0.99), (0.1, 20)])
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

# nicer way to plot with tidy data
# :::::::::::::::: FIG A:::::::::::::::::
columns_a = ['sim', 'trial', 'model', 'p_correct']

df_a = pd.DataFrame(columns=columns_a)
for simnum in range(sim_count):
    rows = [(simnum, x, 'blind', sim_blinds[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_a, data=rows)
    df_a = pd.concat([df_a, temp_df])

    rows = [(simnum, x, 'sights', sim_sights[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_a, data=rows)
    df_a = pd.concat([df_a, temp_df])

fig = sns.lineplot(data=df_a, x='trial', y='p_correct', hue='model')
# fig.set(ylim = (0, 1));
fig.set(ylim = (0.2, 1));


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
