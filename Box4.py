# %% [markdown]

# # Box 4 - parameter recovery in the reinforcement learning model

# Here we have the same task as in box2/fig2, namely the two armed bandit task. We are going to simulate a participant and then recover (fit the model) to the observed behaviour (actions, rewards) to see how close we get to the parameter values used for the simulation. We will do this at least 1000 times to see how well we can expect the recovery to work overall.

# Specifically, one arm has probability 0.2 for reward and the other p=0.8 to receive reward. and we do 1000 trials. this is, of course, a best case scenario type, in a real experiment it would be very unlikely we would get a person to do 1000 trials of this. the algorithm may be able to learn way more quickly than 1000 trials what arm is best, but a good question here is how many trials are needed to be able to recover the parameters with some certainty? let's come back to that question later.

# So, next, to simulate one (ro)bot participant, we use model3, Rescorla-Wagner. this model has two parameters; learning rate alpha and softmax temperature beta. the authors suggest picking random parameter values so let's visualise those values to get a feel for what values to expect.

# ## general imports
# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import seaborn as sns
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from numba import njit

from SimulationFunctions.simulate_M3RescorlaWagner import simulate_M3RescorlaWagner as simulate
from LikelihoodFunctions.lik_M3RescorlaWagner import lik_M3RescorlaWagner as likelihood

# reset look of graphs, gives a decent checkboard background
sns.set()
# %% [markdown]

# ## visualising distributions for parameter values

# ### $\alpha ~ U(0, 1)$

# This one may not really need a visualisation if you're familiar with probability distributions. But I like visualisations and think it's a good practice to confirm things look like expected.
# %%
sns.histplot(np.random.uniform(0, 1, 100000), stat = 'probability')

# %% [markdown]

# ### $\beta ~ Exp(10)$

# Exponential distributions can be more difficult to visualise in your head, especially when it's modified by its "rate parameter" (lambda $\lambda$), in this case, 10.

# %%
sns.histplot(np.random.exponential(10, 100000), stat = 'probability')

# %% [markdown]

# We see in the graph that we can expect most values to be below 40, definitely below 60.

# If our model currently has Q-values for each arm as [0.4, 0.5], with a bigger beta, the more likely the bigger value will be picked. In other words; bigger beta, more greedy behaviour - the higher valued option will always be picked no matter how small the difference between the values. With betas closer to zero, the behaviour becomes more random and even a bigger difference between the option values will mean nothing.

# It's easy to confirm this by playing around with different beta values in the below softmax calculation.

# %%
beta = 0.01
print(np.exp(beta * np.array([0.4, 0.5])) / sum(np.exp(beta * np.array([0.4, 0.5]))))

# %% [markdown]

# ## experiment parameters

# %%
# simulate and fit 1000 times
sim_and_fit_count = 1000
trial_count = 1000  # T in paper's terms
bandit = np.array([0.2, 0.8])  # mu in paper's terms

# %% [markdown]

# ## simulation and fitting

# Now we get to the main bit. Run a simulation with random values for learning rate $\alpha$ and inverse temperature $\beta$. Then fit the model to the data and save the real and fitted parameter values. As seen in Box 3, we run the fitting 10 times with random starting guesses to make sure we find the global optimum.

# Note: 100 `sim_and_fit_count` loops take around 30-40 seconds on my crappy laptop, and 1000 6-7 minutes. We can't speed this up more with numba, as it can't compile scipy's `minimize`, but that's already heavily optimized by very clever people already. What we could do is use the `multiprocessing` module and have each sim and fit loop run on separate thread/cpu core.

# Of course, an even simpler way to speed this up would be to only use one random guess per loop. It actually won't make much of a difference as the same general pattern will show anyway.

# %%
guesses_per_loop = 10
sims_and_fits = []  # data container
bounds = [(0.01, 1), (0.01, 60)]  # alpha, beta

for counter in range(sim_and_fit_count):
    alpha = np.random.uniform(0, 1)
    beta = np.random.exponential(10)
    actions, rewards = simulate(trial_count, bandit, alpha, beta)

    best_llh = 99999  # best loglikelihood will be much lower
    best_result = []
    for _ in range(guesses_per_loop):
        starting_guess = (
            np.random.uniform(0, 1),
            np.random.exponential(10)
        )
        fitresult = minimize(
            likelihood,
            starting_guess,
            args = (actions, rewards),
            bounds = bounds
        )
        if fitresult.fun < best_llh and fitresult.success is True:
            best_result = fitresult
            best_llh = fitresult.fun

    fitalpha = best_result.x[0]
    fitbeta = best_result.x[1]

    sims_and_fits.append([alpha, beta, fitalpha, fitbeta])

# %%
# ## let's plot!

# %%
column_names = ['realalpha', 'realbeta', 'fitalpha', 'fitbeta']
data = pd.DataFrame(columns = column_names, data = sims_and_fits)

# find bad apples. i mean alphas.
data.loc[:, 'badalpha'] = np.abs(data['realalpha'] - data['fitalpha']) > 0.25

sns.scatterplot(data=data, x='realalpha', y='fitalpha', hue='badalpha')

print(f"ratio of bad alphas: {sum(data.badalpha) / sim_and_fit_count}")

# %% [markdown]

# So alpha plot looks pretty good. There's a strong correlation and only 3-4% of the fitted values are "bad".

# Let's check the betas

# %%
# notice we here mark bad alphas, not bad betas
sns.scatterplot(data=data, x='realbeta', y='fitbeta', hue='badalpha')

# %% [markdown]

# As in the paper we see that fits are farily good at $\beta < 10$ and then get increasingly worse. Many of the values hit our upper bound for beta, even though their true value is not really close to 60.

# But it also looks like there are many more bad alphas in the alphas plot than in the betas plot, where did they all go? Let's plot only the betas that are/have badalphas so to speak:

# %%
sns.scatterplot(data=data.query('badalpha == True'), x='realbeta', y='fitbeta')

# %% [markdown]
# ahh it looks like all of them are likely around 0;

# %%
fig = sns.scatterplot(data=data.query('badalpha == True'), x='realbeta', y='fitbeta')
fig.set(xlim=(0,1), ylim=(0,1))

# %% [markdown]
# Now we see it's when realbeta is < 1 that the fitting has issues. This makes sense, because with a very low beta, the behaviour becomes basically random. Any model would have issues fitting well if there is no pattern in the behaviour.

# ## Recovery with fewer trials

# In the above simulations we used 1000 trials for the bandit simulation. But how likely is it that you'll have a human participant doing *one thousand* of these trials? Not very. So, another important thing to look at is how the number of trials impact the parameter recovery quality. We can easily check this by just changing the number of trials above. (The `trial_count` variable)

# What you most likely will see is an increase in "bad" alphas and much worse beta recovery. I don't think the paper brings this up, but it is a highly important aspect. Because you may have to increase the number of trials your human participants ~suffer~ go through in order to have some decent certainty you can fit your model(s).

# ## Confidence for individual fitted parameter values

# General patterns are one thing, but how certain can we be that a single case has been recovered well? The recovery is made on individual level, meaning we fit one pair of parameters for one participant, another pair of parameters for the next participant and so on. In each case, how do we know if it's a "bad" recovery or a "good" one? The simple answer is: we don't. We can only get a general idea of how far, on average, we are from the true parameters.

# Using the data we have, we can investigate what our uncertainty is for each individual parameter recovery. This estimation is, like the previous analyses, dependent on our values for `sim_and_fit_count`, `guesses_per_loop` and `trial_count`. Text/discussion that follows is based on `1000`, `10` and `1000` for these, respectively.

# %%
# calculate distance of each recovered parameter pair from the true values
data.loc[:, 'alphadistance'] = data.realalpha - data.fitalpha
data.loc[:, 'betadistance'] = data.realbeta - data.fitbeta

alphastats = data.alphadistance.describe(percentiles=[0.05, 0.95])

# since each simulation and fit are separate, we treat them as single draws
ci_low, ci_up = stats.norm.interval(
    0.95, loc=alphastats['mean'], scale=alphastats['std']
)

fig = sns.histplot(data.alphadistance, stat = 'probability', kde = True)
fig.set(xlim = (ci_low - 0.1, ci_up + 0.1))
fig.axvline(ci_low)
fig.axvline(ci_up)

# %% [markdown]

# So, in short, we can expect that learning rate alpha will be off by 0.2 in 95% of our cases. Is that acceptable? I can't say, I guess that depends on the experimental task. But having this analysis, we can now go back and simulate/recover to see how much our uncertainty is impacted by different number of trials for example.

# What about uncertainty for $\beta$?

# %%
betastats = data.betadistance.describe()
ci_low, ci_up = stats.norm.interval(
    0.95, loc=betastats['mean'], scale=betastats['std']
)

fig = sns.histplot(data.betadistance, stat = 'probability', kde = True)
fig.set(xlim = (ci_low - 5, ci_up + 5))
fig.axvline(ci_low)
fig.axvline(ci_up)

# %% [markdown]

# This is much more uncertain, but not surprising as we already concluded that beta recovery is best when $1 < \beta < 10$. So if we check only those cases where we had a *fitted* value for beta within that interval (to pretend that we are investigating "real" behavioural data), we get:

# %%
filtered_data = data.query("fitbeta > 1 and fitbeta < 10")

betastats = filtered_data.betadistance.describe()
ci_low, ci_up = stats.norm.interval(
    0.95, loc = betastats['mean'], scale = betastats['std']
)

fig = sns.histplot(filtered_data.betadistance, stat = 'probability', kde = True)
fig.set(xlim = (ci_low - 5, ci_up + 5))
fig.axvline(ci_low)
fig.axvline(ci_up)

# %% [markdown]

# This is actually not too bad. But before we get too pleased with ourselves, let's do the inverse and filter based on the *real* beta values.

# %%
filtered_data = data.query("realbeta > 1 and realbeta < 10")

betastats = filtered_data.betadistance.describe()
ci_low, ci_up = stats.norm.interval(
    0.95, loc = betastats['mean'], scale = betastats['std']
)

fig = sns.histplot(filtered_data.betadistance, stat = 'probability', kde = True)
fig.set(xlim = (ci_low - 5, ci_up + 5))
fig.axvline(ci_low)
fig.axvline(ci_up)

# %% [markdown]

# It can look weird at first glance, but the small distances are due to calculations of $\beta_{real} - \beta_{fit}$ where $\beta_{fit}$ is very large, even hitting the upper bound of 60.

# This looks much worse, and is indeed our worst case scenarios. The upside is that *if* we fit a beta value between 1 and 10 we can be fairly sure it's close to the real value. But if we fit a value outside that range the uncertainty is very large.
