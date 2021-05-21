# %% markdown

# # Box 6 - Improving parameter recovery by modeling unimportant parameters

# TODO: move simulation, likelihood and fitting functions to subfolders when we have decided it's me or the paper being confused.

# This box confuses me. The paper makes it sound like we can modify, for example, Model 3 by adding a bias and thereby capturing some randomness/noise in the simulated data so that the two main parameters - alpha and beta - are recovered better. The title of Box 6 even explicitly says "Example: improving parameter recovery by modeling unimportant parameters."

# But what they do is simulate a biased model - let's call it Model B - and show that a fitting Model B provides better fit than the model without bias, Model M3.

# From the paper:

# > "We then simulated behavior with this model for a range of parameter values and fit the model with the original version of model 3, without the bias, and the modified version of model 3, with the bias."

# ...

# Yes? You simulated with bias so what are we supposed to learn here? That you're simulating a hypothetical person with such a bias? If so, the bias isn't an "unimportant" parameter, it's actually "there" so the biased model is actually a better model, not "unimportant", right?

# I probably misunderstand again, but wouldn't a more interesting test be to simulate with Model M3 and see if Model B helps stabilize recovery of alpha & beta parameters?

# So that's what I've done below..

# alan explains later:

# Yes? You simulated with bias so what are we supposed to learn here? That you're simulating a hypothetical person with such a bias? If so, the bias isn't an "unimportant" parameter, it's actually "there" so the biased model is actually a better model, not "unimportant", right?

# > so you have understood exactly. you need to include the uninteresting process in the modelling in order to get good fits (assuming that the uninteresting process is actually at work in the proess generating the data)

# I probably misunderstand again, but wouldn't a more interesting test be to simulate with Model M3 and see if Model B helps stabilize recovery of alpha & beta parameters?

# > If there is no bias in the true data then including the bias parameter in the model should not help with the model fit -- the average bias parameter you should get out of your fits should be zero. It is an empirical question as to whether including the bias parameter might actually help recover the true alpha and beta when there was actually no bias in the data generating process i.e. the way you did it. Not sure whether this is more interesting -- it is a different question altogether. What did you find?



# ## function and library imports

# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import seaborn as sns
import pandas as pd
from numba import njit, int32
from scipy.optimize import minimize

from SimulationFunctions.simulate_M3RescorlaWagner import simulate_M3RescorlaWagner
from LikelihoodFunctions.lik_M3RescorlaWagner import lik_M3RescorlaWagner

# reset seaborn to get decent looking grid
sns.set()
# %% markdown

# ## Task description and simulation

# The task used in this box is a variant of the standard bandit task used previously. But here we have 10 blocks, each with 50 trials. Every other block we switch which "arm" of the bandit is the better one. In academic parlance this is commonly called a "reversal learning task".

# %%
@njit
def simulate():

    # using similar parameter value generation as matlab code
    alpha = 0.1 + 0.4 * np.random.rand()
    beta = 1 + 8 * np.random.rand()
    trialcount = 50
    block_count = 10

    actions = np.zeros(trialcount * block_count, dtype=int32)
    rewards = np.zeros_like(actions, dtype=int32)

    for block in range(block_count):
        if block % 2 == 0:
            bandit = np.array([0.2, 0.8])
        else:
            bandit = np.array([0.8, 0.2])

        a_block, r_block = simulate_M3RescorlaWagner(
            trialcount, bandit, alpha, beta)

        actions[block * 50:(block + 1) * 50] = a_block
        rewards[block * 50:(block + 1) * 50] = r_block

    return actions, rewards, [alpha, beta]

# %% markdown

# ## Model M3 likelihood adapted to the task

# We have to add the task procedures to our existing likelihood function for Model M3 like we did for the simulation.

# %%
@njit
def llh_m3(parameters, actions, rewards):

    block_count = 10
    total_llh = 0

    for block in range(block_count):
        a_block = actions[block * 50:(block + 1) * 50]
        r_block = rewards[block * 50:(block + 1) * 50]
        total_llh += lik_M3RescorlaWagner(
            parameters, a_block, r_block
        )

    return total_llh

# %% markdown

# ## Model M likelihood

# We also need a new likelihood function for Model M

# According to the paper, we calculate the probability of choosing "left" according to:

# $$p^{left}_t = \frac{1}{1 + exp( \beta (Q^{right}_t−Q^{left}_t − B))}$$

# where $\beta$ is similar to the inverse temperature of softmax and $B$ is the bias. Paper says left but the code below instead follow the matlab code where it's $p^{right}$ or rather `p2`.

# %%
@njit
def llh_m3bias(parameters, actions, rewards):

    alpha = parameters[0]
    beta = parameters[1]
    bias = parameters[2]

    Q = np.array([0.5, 0.5])

    trialcount = len(actions)
    choice_probs = np.zeros(trialcount)

    for trial in range(trialcount):

        if trial % 50 == 0:
            Q = np.array([0.5, 0.5])

        # paper mentions p^left but using matlab code variation
        p2 = 1/(1 + np.exp(beta * (bias + Q[0] - Q[1])))
        probs = np.array([1 - p2, p2])

        choice_probs[trial] = probs[actions[trial]]

        delta = rewards[trial] - Q[actions[trial]]
        Q[actions[trial]] += alpha * delta

    loglikelihood = np.sum(np.log(choice_probs))
    return -loglikelihood

# %% markdown

# ## fitting wrapper functions

# Finally we create simple wrappers around the `scipy.optimize.minimize` function as we did in Box 5.

# %%
def fitm3(actions, rewards):

    best_fit = 9999
    best_params = []
    for startpoint in range(10):
        guess = np.random.rand(2)
        bounds = np.array([(0.01, 0.99), (0.1, 10)])  # alpha, beta
        result = minimize(
            llh_m3, guess, args=(actions, rewards), bounds=bounds)

        if result.fun < best_fit and result.success is True:
            best_fit = result.fun
            best_params = result.x

    return best_fit, best_params


def fitm3bias(actions, rewards):

    best_fit = 9999
    best_params = []
    for startpoint in range(10):
        guess = np.random.rand(3)
        bounds = [
            (0.01, 0.99),  # alpha
            (0.1, 10),     # beta
            (0.01, 0.99)   # bias
        ]
        result = minimize(
            llh_m3bias, guess, args=(actions, rewards), bounds=bounds)

        if result.fun < best_fit and result.success is True:
            best_fit = result.fun
            best_params = result.x

    return best_fit, best_params


# %% markdown

# ## experimental parameters

# %%
simfitcount = 100  # number of simulations and fitting runs

# %% markdown

# ## simulation and fitting runs

# %%
# data container
data = []
# data_m3 = []
# data_m3b = []
# real_params = []

for simfit in range(simfitcount):

    print(f"simfitrun {simfit}")

    actions, rewards, parameters = simulate()

    _, m3_params = fitm3(actions, rewards)
    _, m3b_params = fitm3bias(actions, rewards)
    #
    # data_m3.append(
    #     fitm3(actions, rewards))
    # data_m3b.append(
    #     fitm3bias(actions, rewards))
    # real_params.append(parameters)

    data.append((
        parameters[0],  # real alpha
        parameters[1],  # real beta
        m3_params[0],   # model 3 alpha
        m3_params[1],   # model 3 beta
        m3b_params[0],  # model bias alpha
        m3b_params[1],  # model bias beta
        m3b_params[2]   # model bias bias
    ))

# %% markdown

# ## Plot fitted and real values for Model 3

# ### Real alpha vs fitted alpha for M3

# %%
columns = [
    'real alpha',
    'real beta',
    'm3 alpha',
    'm3 beta',
    'mb alpha',
    'mb beta',
    'mb bias'
]
fit_results = pd.DataFrame(columns = columns, data = data)

fig = sns.regplot(data = fit_results, x = 'real alpha', y = 'm3 alpha')
fig = sns.lineplot(
    x = np.linspace(0, 1, len(fit_results)),
    y = np.linspace(0, 1, len(fit_results)),
    style = True,
    dashes = [(2, 2)],
    legend = False
)
fig.set(xlim = (0, 0.6), ylim = (0, 1), title = "M3 alpha");

# %% markdown

# explain above plot and code

# ### Real beta vs fitted beta for M3

# %%
fig = sns.regplot(data = fit_results, x = 'real beta', y = 'm3 beta')
fig = sns.lineplot(
    x = np.linspace(0, 10, len(fit_results)),
    y = np.linspace(0, 10, len(fit_results)),
    style = True,
    dashes = [(2, 2)],
    legend = False
)
fig.set(xlim = (0, 10), ylim = (0, 10), title = "M3 beta")

# %% markdown

# add some explanatory text here

# ## Plot fitted and real values for Model Bias

# ### Real alpha vs fitted alpha for MB

# %%
fig = sns.regplot(data = fit_results, x = 'real alpha', y = 'mb alpha')
fig = sns.lineplot(
    x = np.linspace(0, 1, len(fit_results)),
    y = np.linspace(0, 1, len(fit_results)),
    style = True,
    dashes = [(2, 2)],
    legend = False
)
fig.set(xlim = (0, 0.6), ylim = (0, 1), title = "MB alpha");

# %% markdown

# ### Real beta vs fitted beta for MB

# %%
fig = sns.regplot(data = fit_results, x = 'real beta', y = 'mb beta')
fig = sns.lineplot(
    x = np.linspace(0, 10, len(fit_results)),
    y = np.linspace(0, 10, len(fit_results)),
    style = True,
    dashes = [(2, 2)],
    legend = False
)
fig.set(xlim = (0, 10), ylim = (0, 10), title = "MB beta")

# %% markdown

# Plots look basically the same for both models, that's not an error. What happens here is that the bias parameter becomes basically zero for all the simulations because, well, we simulated without a bias.

# %%

sns.histplot(fit_results['mb bias'], stat = 'probability')

# %% markdown

# ## Discussion

# I guess we *could* see using the biased model to simulate instead of what we did here as more "interesting" in that it shows that fitting the biased model can account for a participant being biased? I don't know, it feels like I'm missing something here.

# you know, we could do what we did above, and then reach the conclusion that, aha! see, this is what happens when there is no bias in the data! so if we do it again but this time simulate with the biased model, what do we get?
