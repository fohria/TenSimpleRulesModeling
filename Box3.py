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
def simulate_participant():
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

    return data

data = simulate_participant()

# column_names = [
#     'stimulus', 'choice', 'reward', 'setsize', 'block', 'trial'
# ]
# df = pd.DataFrame(columns=column_names, data=data)

# %% [markdown]

# The matlab code has a manual quirk here to check if the performance decreases with increasing setsize. I suspect parameter recovery is better when this pattern is true, but unfortunately there is no discussion in the code or the paper about this. I've chosen not to care about this, but will comment on the problem of recovery after our first heatmap plot.

# Actually, let's check that and see how/if it affects the beta recovery later. Later: actually, yes, with simulation results like this:

# setsize 2 mean: 0.9333333333333335
# setsize 3 mean: 0.8962962962962963
# setsize 4 mean: 0.8944444444444444
# setsize 5 mean: 0.8755555555555556
# setsize 6 mean: 0.8481481481481481

# We get fitted parameters that are very very close to real parameters.

# %%
mean_rewards = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
for block in data:
    setsize = len(np.unique(block[0]))
    mean_rewards[setsize] += np.mean(block[2])
    # print(setsize)

for setsize in mean_rewards:
    print(f"setsize {setsize} mean: {mean_rewards[setsize] / 3}")

# %% [markdown]
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
# cmap = sns.color_palette("colorblind", as_cmap=True)
cmap = 'viridis'
fig = sns.heatmap(
    hot_data, xticklabels = 5, yticklabels = 5, cmap = cmap)

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
def reshape_simdata(data):
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

    return stimuli, choices, rewards, block_indeces

stimuli, choices, rewards, block_indeces = reshape_simdata(data)

# %% [markdown]

# scipy's `minimize` function wants all the parameters in one variable. It also is, as you may have inferred from the name, a minimization function so we need to return the negative loglikelihood.

# We follow the matlab code here in keeping K as a constant, but it can of course be added to the parameters we ask minimize to find.

# %%
@njit
def recover_participant(
    parameters, K, stimuli, choices, rewards, block_indeces):
# def recover_participant(
#     parameters, stimuli, choices, rewards, block_indeces):

    alpha = parameters[0]
    beta = parameters[1]
    rho = parameters[2]
    # K = parameters[3]

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
def fit_participant(stimuli, choices, rewards, block_indeces):
    # alpha, beta, rho
    bounds = [(0.05, 0.99), (1, 30), (0.01, 0.99)]
    # bounds = [(0.01, 0.99), (1, 30), (0.01, 0.99), (2, 6)]
    results = []
    for _ in range(10):
        start_guess = [
            np.random.rand(),       # alpha
            np.random.uniform(20),  # beta
            np.random.rand(),       # rho
            # real_K                  # K
        ]
        result = minimize(
            recover_participant,
            start_guess,
            args = (real_K, stimuli, choices, rewards, block_indeces),
            # args = (stimuli, choices, rewards, block_indeces),
            bounds = bounds
        )
        if result.success is True:
            results.append(result)

    return results

results = fit_participant(stimuli, choices, rewards, block_indeces)
results
# %% [markdown]

# Now we can get the best result from minimize and plot on our heatmap for comparison, and complete figure A.

# %%
# get best minimize result and coordinates for heatmap
best_minimize_index = np.argmin(np.array([res.fun for res in results]))
best_minimize = results[best_minimize_index]
# minimize result's x variable holds [alpha, beta, rho]
best_fit_alpha = best_minimize.x[0]
best_fit_rho = best_minimize.x[2]

x_mark_fit = round((best_fit_alpha - x_min) * x_scale)
y_mark_fit = round((best_fit_rho - y_min) * y_scale)

# cmap = sns.color_palette("colorblind", as_cmap=True)
cmap = 'viridis'
# cmap = 'Spectral'
# plot the heatmap, only print every 5 x/y ticks
fig = sns.heatmap(hot_data, xticklabels=5, yticklabels=5, cmap=cmap)

fig = sns.scatterplot(
    x = [x_mark_real + 0.5],
    y = [y_mark_real + 0.5],
    marker="X",
    color="red"
)
fig = sns.scatterplot(
    x = [x_mark_fit + 0.5],
    y = [y_mark_fit + 0.5],
    marker="*",
    color="blue"
)
fig = sns.scatterplot(
    x = [x_mark_best + 0.5],
    y = [y_mark_best + 0.5],
    marker=".",
    color="black"
)

print(f"red x - real alpha, rho: {real_alpha}, {real_rho}")
print(f"black dot - best brute force alpha, rho: {best_alpha}, {best_rho}")
print(f"blue star - best minfitted alpha, rho: {best_fit_alpha}, {best_fit_rho}")

# %% [markdown]

# With a few iterations of `minimize` and enough parameter combinations for the brute force method, they mostly reach the same result. The brute force method is needed to plot the likelihood surface/heatmap which is good for checking your likelihood function works as expected. When you've done that it may be more convenient to use the `minimize` directly. An additional benefit of `minimize` is that it is usually able to find

# Speaking of checking likelihood function, in some situations our plots above give us results that look off. Meaning they are not at the "max" of the surface. That's not an error, because we are only plotting two out of three estimated parameters.

# So if we check what the likelihoods are in our hot data for the best results and the real values we see the real values indeed has a better likelihood.

# %%
print(f"likelihood for best result: {hot_data.iloc[y_mark_best, x_mark_best]}")
print(f"likelihood for real values: {hot_data.iloc[y_mark_real, x_mark_real]}")

# %% [markdown]

# But if we go back to our dataframe with estimated values for the $\beta$ parameter we can see that the best `loglike` is where it was plotted on the heatmap.

# %%
best_loglike = brute_results.loglike.max()
print(f"best loglike: {best_loglike}")
print(brute_results.query("loglike == @best_loglike"))

#%% [markdown]

# Beta estimation isn't great, but as previously mentioned, this is not something the paper discusses. Running the matlab code gives wildly different distances to the true beta depending on the simulation, so for the next step and figure, be aware that the plot doesn't show how far we are to the real parameters, only how far away we are from the best parameters we can fit.

# ## how many starting guesses before we find global max?

# To create the lineplot in Box3 we run a simulation 10 times, each time using `minimize` to fit 10 times. We save all the results to a dataframe so we can compare the final parameter values found to previous ones and what iteration they were found at.

# %%
from scipy.spatial.distance import euclidean

result_rows = []
for simfit in range(10):

    simdata = simulate_participant()

    stimuli, choices, rewards, block_indeces = reshape_simdata(simdata)

    fit_results = fit_participant(
        stimuli, choices, rewards, block_indeces)

    fit_likelihoods = np.array([result.fun for result in fit_results])
    best_fitindex = np.argmin(fit_likelihoods)
    best_parameters = fit_results[best_fitindex].x

    best_distance_so_far = 999
    for iteration, result in enumerate(fit_results):
        distance_to_best = euclidean(best_parameters, result.x)
        if distance_to_best < best_distance_so_far:
            best_distance_so_far = distance_to_best
        result_rows.append(
            (simfit,
            iteration,
            distance_to_best,
            best_distance_so_far)
        )


columns = [
    'sim iteration',
    'fit iteration',
    'dist to best',
    'best dist so far'
]
result_data = pd.DataFrame(columns = columns, data = result_rows)

sns.lineplot(
    data = result_data,
    x = 'fit iteration',
    y = 'best dist so far'
)

# and log scale, removing the last iteration since it's 0
fig = sns.lineplot(
    data=result_data.query("`fit iteration` != 9"),
    x='fit iteration',
    y='best dist so far'
)
fig.set(yscale="log");
