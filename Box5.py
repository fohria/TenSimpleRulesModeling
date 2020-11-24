# %% [markdown]

# # Box 5 - Confusion matrices in the bandit task

# How do we compare different models to know which one fits our data best?

# We will answer this question by using Models 1-5 from Box 2. For each model, we will simulate it and then fit all 5 models on the data. We should expect, in the best case, that the model used to simulate the data will also fit the best.

# ## Library imports
# %%
# autoreload
%load_ext autoreload
%autoreload 2

import numpy as np
import seaborn as sns
from SimulationFunctions.simulate_M1random import simulate_M1random
from SimulationFunctions.simulate_M2WSLS import simulate_M2WSLS
from SimulationFunctions.simulate_M3RescorlaWagner import simulate_M3RescorlaWagner
from SimulationFunctions.simulate_M4ChoiceKernel import simulate_M4ChoiceKernel
from SimulationFunctions.simulate_M5RWCK import simulate_M5RWCK
from FittingFunctions.fit_all import fit_all

# %% [markdown]

# ## Experiment parameters

# As before, we use a simple two armed bandit and 1000 pulls on each arm for every simulation. As discussed in Box 4, this `trial_count` can be lowered to investigate how well we can recover models with more realistic numbers of trials.

# In the paper, their figure A and C allows for $\beta < 1$. That's equivalent to setting `beta_increase` below to `0`. To generate data where $\beta > 1$, and get figures B and D, simply set `beta_increase` to `1`.

# Running with `simfit_runcount` set to `100` takes ~ 1 min on my laptop. Thanks to having compiled our simulation and likelihood functions with numba, each `simfitrun` loop below went from taking ~1 min to ~800 **milliseconds**. Numba won't always provide such great improvement, but for those likelihood functions where we loop over the actions and rewards, numba is very ... rewarding.

# %%
trial_count = 1000  # T in paper terminology
bandit = np.array([0.2, 0.8])  # mu in paper terminology
beta_increase = 1  # set to 0 for figure A, 1 for figure B
simfit_runcount = 100  # how many times to simulate each model and fit

# %% [markdown]

# ## Simulate participants and fit models

# We create an empty 5x5 matrix to hold our results. Each row represents the simulated model, with columns being the fitted models. Every loop, we first simulate model 1, then fit models 1-5 on the simulation data, and record in the confusion matrix what model had the best fit.

# The `fit_all` function is a short wrapper around individual model specific fitting functions, each using scipy's `minimize` function to estimate parameter values and also returning the BIC value of the model's fit. The [BIC value](https://en.wikipedia.org/wiki/Bayesian_information_criterion) is what's used to decide what model was the best fit.

# $$BIC = -2log\hat{LL} + k_mlog(T)$$

# where $\hat{LL}$ is the likelihood (tiny error in the paper here which states $\hat{LL}$ is the log-likelihood), $k_m$ is the number of parameters for the model and $T$ is the number of trials.

# Code example for Model 3:

# `BIC = -2 * loglikelihood + 2 * np.log(1000)`

# After we have run all the simulations, we divide the values of the confusion matrix with the number of `simfit_runcount` to get ratios.

# %%
confusion_matrix = np.zeros([5, 5])

for simfitrun in range(simfit_runcount):
    print(f"simfitrun {simfitrun}")

    # model 1
    bias = np.random.uniform(0, 1)
    actions, rewards = simulate_M1random(
        trial_count, bandit, bias)

    _, _, best = fit_all(actions, rewards)
    confusion_matrix[0, :] += best

    # model 2
    epsilon = np.random.uniform(0, 1)
    actions, rewards = simulate_M2WSLS(
        trial_count, bandit, epsilon)

    _, _, best = fit_all(actions, rewards)
    confusion_matrix[1, :] += best

    # model 3
    alpha = np.random.uniform(0, 1)
    beta = beta_increase + np.random.exponential(1)
    actions, rewards = simulate_M3RescorlaWagner(
        trial_count, bandit, alpha, beta)

    _, _, best = fit_all(actions, rewards)
    confusion_matrix[2, :] += best

    # model 4
    alpha_c = np.random.uniform(0, 1)
    beta_c = beta_increase + np.random.exponential(1)
    actions, rewards = simulate_M4ChoiceKernel(
        trial_count, bandit, alpha_c, beta_c)

    _, _, best = fit_all(actions, rewards)
    confusion_matrix[3, :] += best

    # model 5
    alpha = np.random.uniform(0, 1)
    beta = beta_increase + np.random.exponential(1)
    alpha_c = np.random.uniform(0, 1)
    beta_c = beta_increase + np.random.exponential(1)
    actions, rewards = simulate_M5RWCK(
        trial_count, bandit, alpha, beta, alpha_c, beta_c)

    _, _, best = fit_all(actions, rewards)
    confusion_matrix[4, :] += best

#%% [markdown]

# ## Plot confusion matrix

# %%
cmap = sns.color_palette("colorblind", as_cmap = True)
# cmap = 'viridis'
fig = sns.heatmap(
    confusion_matrix / simfit_runcount, annot = True, cmap = cmap)
fig.set(xlabel = "fit model", ylabel = "simulated model");

# %% [markdown]

# ## Plot inverse confusion matrix

# From the paper:

# > "The inversion matrix provides easier interpretation of fitting results when the true model is unknown. For example, the confusion matrix indicates [FigureA, betaincrease = 0] that M1 is always perfectly recovered, while M5 is only recovered 30% of the time. By contrast, the inversion matrix shows that if M1 is the best fitting model, our confidence that it generated the data is low (54%), but if M5 is the best fitting model, our confidence that it did generate the data is high (97%)"

# > "Note that this measure, which we term the ‘inversion matrix’ to distinguish it from the confusion matrix, is not the same as the confusion matrix unless model recovery is perfect."

# I'm not sure i understand the "unless model recovery is perfect", what is the inversion matrix really?

# %%
inverse_confmatrix = np.zeros_like(confusion_matrix)
for column in range(inverse_confmatrix.shape[1]):
    inv_column = confusion_matrix[:, column] / np.sum(confusion_matrix[:, column])
    inverse_confmatrix[:, column] = inv_column

fig = sns.heatmap(inverse_confmatrix, annot = True, cmap = cmap)
fig.set(xlabel = "fit model", ylabel = "simulated model");

# %% [markdown]

# ## Discussion

# From the paper:

# > "As with parameter recovery, we believe that the best approach is to match the range of the parameters to the range seen in your data, or to the range that you expect from prior work."

# So, I know what we're doing is fiddly and messy, but is the advice to look at your experimental data really the best? it feels like there's an issue methodologically there, unless the data you're talking about is explicitly a pilot experiment to explore what parameter ranges your human participant data gives you?
