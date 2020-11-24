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
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from SimulationFunctions.simulate_box7 import simulate_blind, simulate_sights
from LikelihoodFunctions.lik_box7 import llh_sights, posterior

# %% [markdown]

# ## Description of models

# Both models use a "basic" reinforcement learning algorithm like Model 3 that we have used in previous boxes. But in the previous task used - the two armed bandit - there was never a state, only actions. Well, I guess you could say there was an implicit single ever-lasting state. That's in essence what the "blind" model does; it has no concept of different states but tries to learn the values of each action anyway.

# In appendix 5 of the paper it's further described that we have two learning rates, $\alpha_{pos}$ and $\alpha_{neg}$, for positive reward prediction error and negative prediction error, respectively.

# $\alpha_{neg}$ is set to 0 for all simulations and same for the likelihood functions in the Matlab code, so we will do the same here. In practice, we thus only update action values if the prediction error is positive, i.e. if the reward received is larger than predicted.

# Choices are again made by softmax, using $\beta$ as our inverse temperature.

# As in previous boxes, we import the simulation and likelihood functions and here present the main loop and its utility functions.

# %% [markdown]

# `fitRL` wraps our likelihood and posterior functions. The posterior function is used to get the likelihood values for figure B so that we can plot them. We also run the minimization 10 times to find the best global optimum.

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

# ## collating the data
# %%
combined_columns = ['simcount', 'trial', 'model', 'p_correct', 'figure']
all_sims = pd.DataFrame(columns = combined_columns)

for simnum in range(sim_count):
    # figure a
    rows_sim_blinds = [
        (simnum, x, 'blind', sim_blinds[simnum][x], 'a')
        for x in range(15)
    ]
    rows_sim_sights = [
        (simnum, x, 'sights', sim_sights[simnum][x], 'a')
        for x in range(15)
    ]
    # figure b
    rows_like_blinds = [
        (simnum, x, 'blind', like_blinds[simnum][x], 'b')
        for x in range(15)
    ]
    rows_like_sights = [
        (simnum, x, 'sights', like_sights[simnum][x], 'b')
        for x in range(15)
    ]
    # figure c
    rows_simfit_blinds = [
        (simnum, x, 'blind', simfit_blind[simnum][x], 'c')
        for x in range(15)
    ]
    rows_simfit_sights = [
        (simnum, x, 'sights', simfit_sight[simnum][x], 'c')
        for x in range(15)
    ]
    simrun_data = rows_sim_blinds + rows_sim_sights + rows_like_blinds + rows_like_sights + rows_simfit_blinds + rows_simfit_sights
    simrun_df = pd.DataFrame(columns = combined_columns, data = simrun_data)
    all_sims = pd.concat([all_sims, simrun_df])

df_combo
all_sims
# %% [markdown]
# ## Figure A

# %%
sns.set(rc={"figure.figsize": (3, 3), "figure.dpi": 100})

fig = sns.lineplot(
    data = all_sims.query("figure == 'a'"),
    x = 'trial',
    y = 'p_correct',
    hue = 'model'
)
fig.set(ylim = (0.2, 1));

# %% [markdown]
# ## Figure B
# %%
fig = sns.lineplot(
    data = all_sims.query("figure == 'b'"),
    x = 'trial',
    y = 'p_correct',
    hue = 'model'
)
fig.set(ylim = (0.2, 1), ylabel = "likelihood of choice");

# %% [markdown]
# # Figure C
# %%
fig = sns.lineplot(
    data = all_sims.query("figure == 'c'"),
    x = 'trial',
    y = 'p_correct',
    hue = 'model'
)
fig.set(ylim = (0.2, 1));

# %% [markdown]

# ## Plotting all together

# It's nice to see the plots next to each other, but to do this we need to have the same column names which may or may not be more confusing. Especially for the middle plot, we can't get seaborn to change its y-axis name as it should be likelihood and not p(correct). If you happen to know how to do this please let me know!

# %%
grid = sns.FacetGrid(all_sims, col = 'figure', hue = 'model')
grid.map(sns.lineplot, 'trial', 'p_correct')
grid.add_legend()
grid.set_xlabels("time step")
grid.set(ylim=(0.2, 1), xlim=(-1, 15), xticks=[0, 5, 10, 14])
grid.set_ylabels("p(correct)")
# grid.despine()
axes = grid.axes.flatten()
axes[0].set_title("A: 'subject' learning curves")
axes[0].set_ylabel("p(correct)")
axes[1].set_title("B: likelihood of \nstate-based RL model")
axes[1].set_ylabel("likelihood of choice")
axes[2].set_title("C: simulated learning curves \nfrom state-based RL")
axes[2].set_ylabel("p(correct)");
