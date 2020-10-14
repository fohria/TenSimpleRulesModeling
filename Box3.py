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
alphas = np.arange(0.05, 1, 0.2)
betas = np.arange(1, 30, 10)
rhos = np.arange(0.05, 1, 0.2)
Ks = np.arange(2, 7)

# finer grid
alphas = np.arange(0.05, 0.51, 0.01)
betas = np.append([1], np.arange(4, 20, 2))
rhos = np.arange(0.5, 1.0, 0.01)
Ks = np.arange(2, 7)

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

        data.append([stimuli, choices, rewards])

        # for row in rows:
        #     data.append(row)

        # data.append(rows)

# column_names = [
#     'stimulus', 'choice', 'reward', 'setsize', 'block', 'trial'
# ]
# df = pd.DataFrame(columns=column_names, data=data)

# %% [markdown]

# not sure saving as dataframe here is best choice, but above is fairly clean anyway. i think the next step with likelihoods is what took long time earlier but probably due to likelihood function taking long time so lets see.

# ## calculate likelihood for all parameter combinations

# %%
from LikelihoodFunctions.lik_RLWM import likelihood_RLWM_block

loglikes = []

for alpha, beta, rho, K in product(alphas, betas, rhos, Ks):

    loglike_allblocks = 0
    for block in range(len(data)):

        blockdata = np.array(data[block])  # numba wants numpy array

        loglike_allblocks += likelihood_RLWM_block(
            alpha, beta, rho, K, blockdata)

    loglikes.append(
        (alpha, beta, rho, K, loglike_allblocks))

fit_results = pd.DataFrame(
    columns = ['alpha', 'beta', 'rho', 'K', 'loglike'],
    data = loglikes
)

# filter data to get same value ranges as in paper's plot
filtered_results = fit_results.query("alpha <= 0.5 and rho >= 0.5")


# hot_data = filtered_results[[
hot_data = fit_results[[
    'alpha', 'rho', 'loglike'
]].pivot_table(
    values = 'loglike',
    # index = 'alpha',
    # columns = 'rho',
    index = 'rho',
    columns = 'alpha',
    aggfunc = 'mean'
)



sns.heatmap(hot_data)
