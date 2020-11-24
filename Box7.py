# %% [markdown]

# # Box 7 - Model validation where the fit model performs too well

# TODO: put functions in subfolders

# TODO: save to dataframe while simfitting, can then plot immediately without creating dataframes individually

# This one is a bit tricky. We have two models: blind and state-based (which I will sometimes call "sighted"). So far so good. What can be confusing (at least it was to me) is the difference in what is being plotted in Figure A, B and C. So based on the plots from the paper, here's what happens:

# **Figure A:** We simulate the behaviour of both models and plot their performance. The two lines represent two different models, each with its own data.

# **Figure B:** This shows how we model the two participants with the state-based (sighted) model. In other words, we fit the sighted model to both blind and sighted data. One model, two sets of data. Interestingly, it seems the sighted model has a higher likelihood for explaining the blind data than the sighted data.

# **Figure C:** If we use the fitted parameters found in figure B and *simulate* using the sighted model, we find that the behaviour produced (in the form of learning curves) looks like the sighted behaviour in figure A. But when simulating the sighted model with the parameters fit to *blind data* it produces behaviour that doesn't look like the behaviour in figure A.

# In other words, it may be *theoretically* correct that one model has higher likelihood than the true one (figure b), we should always test this in *practice* to confirm if the behaviour is the same (figure c vs figure a).

# Got it? Cool :) Let's describe the task we will use for this experiment!

# ## The experimental task

# Imagine there are three different pictures that can be shown to you, and you have to learn which of three buttons to push depending on the picture. That's the "sighted" model. For the "blind" model, imagine that you are, well, blind. You get told when it's time to push a button, maybe you will learn the correct pattern, maybe you won't. Sounds like a horrifying task now that I put it that way.

# The paper describes the basic rules of the task, where we have three stimuli $s_1, s_2, s_3$ and three actions $a_1, a_2, a_3$.

# $a_1$ is the correct choice for $s_1, s_2$

# $a_3$ is the correct choice for $s_3$

# $a_2$ does nothing

# From the matlab code we see that we have 10 blocks, and learning values are reset at the start of each block. Every block has 45 trials.

# Since python has 0-index we will use that so in the following code we have three stimuli $s_0, s_1, s_2$ and three actions $a_0, a_1, a_2$.

# $a_0$ is the correct choice for $s_0, s_1$

# $a_2$ is the correct choice for $s_2$

# $a_1$ does nothing

# ## Library and function imports

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

# %% [markdown]

# ## Description of models

# Both models use a "basic" reinforcement learning algorithm like Model 3 that we have used in previous boxes. But in the previous task used - the two armed bandit - there was never a state, only actions. Well, I guess you could say there was an implicit single ever-lasting state. That's in essence what the "blind" model does; it has no concept of different states but tries to learn the values of each action anyway.

# In appendix 5 of the paper it's further described that we have two learning rates, $\alpha_{pos}$ and $\alpha_{neg}$, for positive reward prediction error and negative prediction error, respectively.

# $\alpha_{neg}$ is set to 0 for all simulations and same for the likelihood functions in the Matlab code, so we will do the same here. In practice, we thus only update action values if the prediction error is positive, i.e. if the reward received is larger than predicted.

# Choices are again made by softmax, using $\beta$ as our inverse temperature.

# ## Simulation functions

# %%
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

# %%
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

# %% [markdown]

# ## Likelihood function

# We won't fit the blind model so we only need a likelihood function for the sighted version.

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

# %% [markdown]

# The posterior function is used to get the likelihood values for figure B

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


# %% [markdown]

# `fitRL` wraps our likelihood and posterior functions. We also run the minimization 10 times to find the best global optimum.

# %%
def fitRL(states, actions, rewards):

    bounds = [(0.01, 0.99), (0.1, 20)]  # alpha, beta
    best = 9999
    best_parms = []

    for _ in range(10):
        guess = (np.random.rand(), np.random.rand() * 10)  # alpha, beta
        result = minimize(
            llh_sights, guess, args=(states, actions, rewards), bounds=bounds)
        if result.fun < best:
            best_parms = result.x

    likes = posterior(best_parms, states, actions, rewards)

    return best_parms, likes

# %% [markdown]

# `summarize_per_stimuli` averages performance results across each stimuli. We have 45 trials in each block, and stimuli are presented in order, so we get $45 / 3 = 15$ trials on the x-axis for plotting.

# %%
def summarize_per_stimuli(series):

    stim1 = np.array([series[np.arange(block * 45, (block + 1) * 45, 3)] for block in range(10)])
    stim2 = np.array([series[np.arange(block * 45 + 1, (block + 1) * 45, 3)] for block in range(10)])
    stim3 = np.array([series[np.arange(block * 45 + 2, (block + 1) * 45, 3)] for block in range(10)])

    return np.mean((stim1 + stim2 + stim3) / 3, axis=0)


# %% [markdown]
# ## Simulate, fit and simulate again!

# Now for our main loop. We simulate the blind model, then fit the sighted model to this data and save the fitting result. That fitting result - the estimated values for alpha and beta - is then used to simulate the sighted model.

# Then we simulate the sighted model, fit the sighted model to that data and finally simulate the sighted model again using the estimated parameter values we just got.

# ### Experimental parameters

# `sim_count` defines how many loops of sim-fitting we will do.

# Default simulation parameters are the specified values from the paper so we get similar performance curves in figure A for both models. Matlab code has slightly enlarged ranges for these values, still producing similar learning curves, that can be uncommented for some extra experimentation.

# %%
sim_count = 100  # how many loops in total

# np.tile is not supported by numba but is the same every loop anyway
stimuli = np.tile(np.array([0, 1, 2], dtype=int), int(45 / 3))
# fig a data
sim_blinds = []
sim_sights = []
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

    # state-based/sighted model
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


# %% [markdown]
# ## Figure A
# %%
sns.set(rc={"figure.figsize": (3, 3), "figure.dpi": 100})
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
fig.set(ylim = (0.2, 1));


# %% [markdown]
# ## Figure B
# %%
columns_b = ['simcount', 'trial', 'data_model', 'p_like']

df_b = pd.DataFrame(columns=columns_b)
for simnum in range(sim_count):
    rows = [(simnum, x, 'blind', like_blinds[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_b, data=rows)
    df_b = pd.concat([df_b, temp_df])

    rows = [(simnum, x, 'sights', like_sights[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_b, data=rows)
    df_b = pd.concat([df_b, temp_df])

fig = sns.lineplot(data=df_b, x='trial', y='p_like', hue='data_model')
fig.set(ylim = (0.2, 1));


# %% [markdown]
# # Figure C
# %%
columns_c = ['simcount', 'trial', 'sim_model', 'p_correct']

df_c = pd.DataFrame(columns=columns_c)
for simnum in range(sim_count):
    rows = [(simnum, x, 'blind', simfit_blind[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_c, data=rows)
    df_c = pd.concat([df_c, temp_df])

    rows = [(simnum, x, 'sights', simfit_sight[simnum][x]) for x in range(15)]
    temp_df = pd.DataFrame(columns=columns_c, data=rows)
    df_c = pd.concat([df_c, temp_df])

fig = sns.lineplot(data=df_c, x='trial', y='p_correct', hue='sim_model')
fig.set(ylim = (0.2, 1));


# %% [markdown]

# ## Plotting all together

# nice to see plots next to each other, but to do this we need to have the same column names which may or may not be more confusing. especially for the middle plot, we can't get seaborn to change its y-axis name as it should be likelihood and not p(correct).

# %%
df_a = df_a.rename(columns={'sim': 'simcount'})
df_a['fig'] = 'a'

df_b = df_b.rename(columns={'data_model': 'model'})
df_b = df_b.rename(columns={'p_like': 'p_correct'})
df_b['fig'] = 'b'

df_c = df_c.rename(columns={'sim_model': 'model'})
df_c['fig'] = 'c'

df_combo = df_a.append(df_b).append(df_c)

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
