# %% [markdown]

# # Box3 - contending with multiple local maxima

# This box was confusing to me. The task and model used are different from the task and models in Box2 and Boxes 4-7. The text in the main paper says there are two parameters, learning rate $\alpha$ and $\rho$ which controls working memory influence. However, appendix 4 mentions four parameters and refers to a paywalled paper (Collins & Frank, 2012) for a complete description of the task used. When going through the Matlab code we see there are indeed four parameters, but one of these, $K$, is kept constant for simulations and recovery. So there are in fact three parameters we use for simulation and recovery here.

# I mention this because I *think* I've gotten the task and model right, but there may be some discrepancies left. However, this shouldn't matter for the principles shown.

# ## Experimental task

# Our simulated participants will partake in a task where they see a picture (stimulus) and through trial-and-error have to learn which of three buttons (our actions or choices) to push to receive a reward. The experiment has multiple blocks, each with a different number of stimuli ranging from 2 to 6 (`setsize` variable below). Every block is repeated three times, so we have a total of $5 * 3 = 15$ blocks. Within a block, each stimuli is repeated 15 times giving us a total of $(2 + 3 + 4 + 5 + 6) * 15 = 300$ trials, and including the block repeats we thus have $300 * 3 = 900$ trials in total.

# ## Learning model

# This model uses a mix of reinforcement learning (RL) and working memory (WM). So we have the standard RL with values for each combination of stimuli (state) and action that's updated every trial like:

# $$Q(state, action) = Q(state, action) + \alpha * (reward - Q(state, action))$$

# and then we have similar state, action values for WM that are just set to the reward received:

# $$WM(state, choice) = reward$$

# Choices are made using softmax with inverse temperature $\beta$ for RL:

# $$p(action) = \frac{e^{\beta * Q(state)}}{\sum{e^{\beta * Q(state)}}}$$

# and for WM the same softmax is used, but using a constant of $50$ instead of $\beta$ which essentially creates a fully "greedy" behaviour, almost always picking the highest valued action.

# The final choice probabilities are calculated with a mixture policy:

# $$p(action) = (1 - w) * p_{RL} + w * p_{WM}$$

# where $w = \rho * min(1, \frac{K}{ns})$, where $ns$ is number of stimuli and $K$ can be seen as scaling the mixture weight in proportion to working memory capacity vs number of stimuli.

# ## what we will do

# we will first simulate one participant, then use that data to plot a likelihood heatmap for many combinations of the four parameters.

# ## import libraries and functions
# %%
# autoreload files in ipython
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, int32
from scipy.optimize import minimize
from itertools import product

from SimulationFunctions.simulate_RLWM import simulate_RLWM_block
from LikelihoodFunctions.lik_RLWM import likelihood_RLWM_block

# %% [markdown]

# ## experimental parameters

# %%
# simulation parameters
real_alpha = 0.1
real_beta = 10
real_rho = 0.9
real_K = 4

# range of parameter values to calculate likelihoods for
# alphas = np.arange(0.05, 1, 0.2)
# betas = np.arange(1, 30, 10)
# rhos = np.arange(0.05, 1, 0.2)
# Ks = np.arange(2, 7)

# finer grid
alphas = np.arange(0.05, 0.51, 0.01)
betas = np.append([1], np.arange(4, 20, 2))
rhos = np.arange(0.5, 1.0, 0.01)
# Ks = np.arange(2, 7)
Ks = np.array([4])  # matlab code keeps this constant

# %% [markdown]

# ## simulation

# the imported simulation function simulates one block of the task, so we construct the blocks here for clarity.

# %%
data = []
block = -1  # to match 0-indexing and keep track of block

for _ in range(3):  # 3 repetitions of each setsize
    for setsize in range(2, 7):  # up to but not including 7

        block += 1

        stimuli, choices, rewards = simulate_RLWM_block(
            real_alpha, real_beta, real_rho, real_K, setsize)

        # rows = [
        #     (stimulus, choice, reward, setsize, block, trial)
        #     for trial, (stimulus, choice, reward)
        #     in enumerate(zip(stimuli, choices, rewards))
        # ]
        #
        # for row in rows:
        #     data.append(row)

        data.append([stimuli, choices, rewards])  #single line

        # data.append(rows)

# column_names = [
#     'stimulus', 'choice', 'reward', 'setsize', 'block', 'trial'
# ]
# df = pd.DataFrame(columns=column_names, data=data)

# %% [markdown]

# The matlab code has a manual quirk here to check if the performance decreases with increasing setsize. I suspect parameter recovery is better when this pattern is true, but unfortunately there is no discussion in the code or the paper about this. I've chosen not to care about this, but will comment on the problem of recovery after our first heatmap plot.

# ## calculate likelihood for all parameter combinations

# we will now go through all the combinations of alpha, beta, rho and K we have and for each combination calculate the summed loglikelihood of all actions taken.

# what this means is that we go through the choices from the simulation, and for each trial calculate the probability of that choice being made, based on a specific set of parameter values. So for parameter values very far from the true parameter values we used to run the simulation, we should get a much lower "score" (likelihood) than for parameter values very close to the true values.

# %%
loglikes = []

for alpha, beta, rho, K in product(alphas, betas, rhos, Ks):

    loglike_allblocks = 0
    for block in range(15):

        # blockdata = np.array(df.query('block == @block'))
        blockdata = np.array(data[block])  # numba wants numpy array

        loglike_allblocks += likelihood_RLWM_block(
            alpha, beta, rho, K, blockdata)

    loglikes.append(
        (alpha, beta, rho, K, loglike_allblocks))


brute_results = pd.DataFrame(
    columns = ['alpha', 'beta', 'rho', 'K', 'loglike'],
    data = loglikes
)

# %% [markdown]

# I'm allergic to importing matplotlib just to do seemingly simple things such as modifying the labels on the x/y axis. Seaborn works okay for most things without such sillyness, but heatmap is a bit too complicated so it's cumbersome to customize. So we have to manually round the parameter values (because python often generates 7.000000000000001 instead of 7.0), and then tell `sns.heatmap` we only want every fifth tick label.

# If you happen to know how I can give heatmap a list of y labels like `[0.5, 0.6, 0.7, 0.8, 0.9]` and it will use that automagically, please let me know!

# A nice and convenient thing about seaborn though, is that when you run several plot commands in succession they'll automatically overlay on the same figure, and we exploit that to mark the 'real' parameters used for the simulation (the red x marks the spot!) and the best parameters found (the black dot).

# ## create pivot table and plot likelihood surface

# %%
brute_results.alpha = brute_results.alpha.apply(lambda x: round(x, 2))
brute_results.rho = brute_results.rho.apply(lambda x: round(x, 2))

hot_data = brute_results[[
    'alpha', 'rho', 'loglike'
]].pivot_table(
    values = 'loglike',
    index = 'rho',
    columns = 'alpha',
    aggfunc = 'mean'
)

# setup the look of our plot
sns.set(rc={
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "lines.markersize": 10,
    "font.size": 10
})

# heatmap coordinates are based on column indeces
x_min = hot_data.columns.min()
x_max = hot_data.columns.max()
x_scale = (len(hot_data.columns) - 1) / (x_max - x_min)
y_min = hot_data.index.min()
y_max = hot_data.index.max()
y_scale = (len(hot_data.index) - 1) / (y_max - y_min)

x_mark_real = round((real_alpha - x_min) * x_scale)
y_mark_real = round((real_rho - y_min) * y_scale)

# get the best result and its coordinates for the heatmap
best_index = brute_results.loglike.idxmax()
best_result = brute_results.iloc[best_index, :]
best_alpha = best_result['alpha']
best_rho = best_result['rho']

x_mark_best = round((best_alpha - x_min) * x_scale)
y_mark_best = round((best_rho - y_min) * y_scale)

# plot the heatmap, only print every 5 x/y ticks
fig = sns.heatmap(hot_data, xticklabels=5, yticklabels=5)

fig = sns.scatterplot(
    x = [x_mark_real + 0.5],  # put mark in middle of the heatmap box
    y = [y_mark_real + 0.5],
    marker="X",
    color="red"
)
fig = sns.scatterplot(
    x = [x_mark_best + 0.5],
    y = [y_mark_best + 0.5],
    marker=".",
    color="black"
)
print(f"real alpha, rho: {real_alpha}, {real_rho}")
print(f"best alpha, rho: {best_alpha}, {best_rho}")

# %% [markdown]

# Note that seaborn/matplotlib puts the upmost part of the y-axis labels where their corresponding boxes are, so it might look a bit off vertically.

# Now, in the case of our brute force search above, we found that the best likelihood wasn't actually found at the real/simulated parameter values. If we re-run the simulation and brute force again we may or may not get closer.

# This is unfortunately something we have to accept; the problem of recovering parameter values is a statistical one and therefore we are going to get slightly different results every time. We will see in the next box, Box4, how to check the overall performance of parameter recovery.

# %% [markdown]

# ## using optimization algorithms to find parameters

# Instead of using the brute force search above, SciPy has a nice optimization package with a function called `minimize`. We can feed our likelihood function into `minimize` and have it find the best fit for us.

# The downside of this function is that it can get stuck in local minima, so it's a good idea to give it a bunch of random starting points. The brute force method doesn't have this problem as it will methodically go through the entire parameter space with the "resolution" (number of parameter combinations) we have chosen.

# The upside is it can often be much quicker, especially if we create an optimized likelihood function with numba.

# ## optimized likelihood function

# %% [markdown]

# First we need to reshape our simulation data, because numpy/numba doesn't like "ragged" arrays, meaning arrays of different lengths.

# Somewhat ugly but can be cleaned if we save the simulation data smarter.

# %%

trials = 0
stimuli = []
choices = []
rewards = []
block_indeces = []

for block in data:
    # stimuli, choices, rewards
    for trial in range(len(block[0])):
        stimuli.append(block[0][trial])
        choices.append(block[1][trial])
        rewards.append(block[2][trial])
        trials += 1
    block_indeces.append(trials)

stimuli = np.array(stimuli)
choices = np.array(choices)
rewards = np.array(rewards)
block_indeces = np.array(block_indeces)

# %% [markdown]

# scipy's `minimize` function wants all the parameters in one variable. It also is, as you may have inferred from the name, a minimization function so we need to return the negative loglikelihood.

# We follow the matlab code here in keeping K as a constant, but it can of course be added to the parameters we ask minimize to find.

# %%
@njit
def recover_participant(
    parameters, K, block_indeces, stimuli, choices, rewards):

    alpha = parameters[0]
    beta = parameters[1]
    rho = parameters[2]

    loglike_allblocks = 0

    low_index = 0
    for index in block_indeces:

        high_index = index
        blocksize = high_index - low_index
        # numba wants an empty array to fill
        blockdata = np.empty((3, blocksize), dtype=int32)
        blockdata[0] = stimuli[low_index:high_index]
        blockdata[1] = choices[low_index:high_index]
        blockdata[2] = rewards[low_index:high_index]

        loglike_allblocks += likelihood_RLWM_block(
            alpha, beta, rho, K, blockdata)
        low_index = index

    return -loglike_allblocks


# %% [markdown]

# Now we run minimize 10 times and save the results to then extract the best result. It's almost always a good practice to provide bounds to the method, otherwise we may get over/underflow errors. This can happen anyway, but luckily `minimize` tells us if the operation was a success or not which is why we have that additional check before appending the result.

# In the below call to minimize, `start_guess` will automagically become the `parameters` in the function `recover_participant` we defined above, while the `args` become the rest of the arguments to the function, in order.

# %%
# alpha, beta, rho, K
# bounds = ((0.01, 0.99), (1, 30), (0.01, 0.99), (2, 6))
bounds = [(0.01, 0.99), (1, 30), (0.01, 0.99)]
results = []
for _ in range(10):
    start_guess = [
        np.random.rand(),       # alpha
        np.random.uniform(20),  # beta
        np.random.rand(),       # rho
        #real_K
    ]
    result = minimize(
        recover_participant,
        start_guess,
        # args=(real_K, data),  # data from the simulation earlier
        # below will not work because it cant convert dtype object
        # args=(real_K, np.array(data),  # data from the simulation earlier
        args = (real_K, block_indeces, stimuli, choices, rewards),
        bounds = bounds
    )
    if result.success is True:
        results.append(result)

# %% [markdown]

# Now we can get the best result from minimize and plot on our heatmap for comparison, and complete figure A.

# %%
# get best minimize result and coordinates for heatmap
best_minimize_index = np.argmin(np.array([res.fun for res in results]))
best_minimize = results[best_minimize_index]
# minimize result's x variable holds [alpha, beta, rho]
best_fit_alpha = best_minimize.x[0]
best_fit_rho = best_minimize.x[2]

x_min = hot_data.columns.min()
x_max = hot_data.columns.max()
x_scale = (len(hot_data.columns) - 1) / (x_max - x_min)
y_min = hot_data.index.min()
y_max = hot_data.index.max()
y_scale = (len(hot_data.index) - 1) / (y_max - y_min)

best_fit_x_mark = round((best_fit_alpha - x_min) * x_scale)
best_fit_y_mark = round((best_fit_rho - y_min) * y_scale)




# get the best result and its coordinates for the heatmap
best_index = brute_results.loglike.idxmax()
best_result = brute_results.iloc[best_index, :]
best_alpha = best_result['alpha']
best_rho = best_result['rho']
x_mark_best = np.argwhere(hot_data.columns == best_alpha).flatten()[0]
y_mark_best = np.argwhere(hot_data.index == best_rho).flatten()[0]

# plot the heatmap, only print every 5 x/y ticks
fig = sns.heatmap(hot_data, xticklabels=5, yticklabels=5)

fig = sns.scatterplot(
    x = [x_mark_real + 0.5],  # put mark in middle of the heatmap box
    y = [y_mark_real + 0.5],
    marker="X",
    color="red"
)
fig = sns.scatterplot(
    x = [x_mark_best + 0.5],
    y = [y_mark_best + 0.5],
    marker=".",
    color="black"
)
print(f"real alpha, rho: {real_alpha}, {real_rho}")
print(f"best alpha, rho: {best_alpha}, {best_rho}")
