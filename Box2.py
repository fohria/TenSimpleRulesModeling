# %% markdown

# # Box2 - Simulate behaviour in the bandit task

# TODO: maybe move this text to a Box1 file where we present the models?

# The bandit task is fairly straight-forward. Imagine yourself in a casino, playing the slot machines - one armed bandits. Every time you pull an arm, there's some probability of getting a payout (reward). Now, say there are two one armed bandits (or, as ~some~ I would put it, one bandit with two arms); can we figure out which bandit/arm has a higher chance of reward?

# What we will do here is simulate "artificial" participants playing this game (performing this task in academic speak). We will have five different "types" of players - five different models - each using a specific strategy to choose what arm/bandit to pull.

# From the paper:

# > More specifically, we consider the case in which a participant makes a series of T choices between K slot machines, or ‘one-armed bandits’, to try to maximize their earnings. If played on trial $t$, each slot machine, $k$, pays out a reward, $r_t$, which is one with reward probability, $\mu^k_t$, and otherwise $0$. The reward probabilities are different for each slot machine and are initially unknown to the subject. In the simplest version of the task, the reward probabilities are fixed over time.

# > The three experimental parameters of this task are: the number of trials, $T$, the number of slot machines, $K$, and the reward probabilities of the different options, $\mu^k_t$, which may or may not change over time. The settings of these parameters will be important for determining exactly what information we can extract from the experiment. In this example, we will assume that $T=1000$, $K=2$, and that the reward probabilities are $\mu^1_t=0.2$ for slot machine 1 and $\mu^2_t=0.8$ for slot machine 2.

# ## import libraries and functions

# %%
# autoreload of imported functions when using ipython/jupyter
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product

from SimulationFunctions.simulate_M1random import simulate_M1random
from SimulationFunctions.simulate_M2WSLS import simulate_M2WSLS
from SimulationFunctions.simulate_M3RescorlaWagner import simulate_M3RescorlaWagner
from SimulationFunctions.simulate_M4ChoiceKernel import simulate_M4ChoiceKernel
from SimulationFunctions.simulate_M5RWCK import simulate_M5RWCK
from AnalysisFunctions.calculate_WSLS import winstay_losestay

# %% markdown

# ## define experimental parameters

# %%
trialcount = 1000  # number of trials, T in the paper
bandit = np.array([0.2, 0.8])  # mu in the paper
simulation_count = 110  # Nrep in matlab code, a.k.a. number of participants
all_sims = [{'actions': [], 'rewards': []} for _ in range(6)]
# %% markdown

# I use the concept of one bandit with many arms. So each item in the `bandit` variable represents an arm, and the item values represents the probability of reward for pulling that arm.

# I prefer to think of repeated simulations of the same model as several participants of the same "type", so instead of `Nrep` for number of repetitions we have `simulation_count` for number of artificial/fake/simulated participants.

# We also setup `all_sims` to hold our results for the 5 models. To make the model numbers match with those of the paper, there's a dummy 0 entry in this list, so model1 has index 1, model2 has index2 etc.

# ## Model 1 : random with bias

# %%
for _ in range(simulation_count):  # _ used to indicate unused variable
    bias = 0.5
    actions, rewards = simulate_M1random(trialcount, bandit, bias)
    all_sims[1]['actions'].append(actions)
    all_sims[1]['rewards'].append(rewards)

# %% [markdown]
# ## Model 2 : Win-Stay Lose-Shift

# %%
for _ in range(simulation_count):
    epsilon = 0.1
    actions, rewards = simulate_M2WSLS(trialcount, bandit, epsilon)
    all_sims[2]['actions'].append(actions)
    all_sims[2]['rewards'].append(rewards)

# %% [markdown]
# ## Model 3 : Rescorla-Wagner

# %%
for _ in range(simulation_count):
    alpha = 0.1
    beta = 5
    actions, rewards = simulate_M3RescorlaWagner(
        trialcount, bandit, alpha, beta)
    all_sims[3]['actions'].append(actions)
    all_sims[3]['rewards'].append(rewards)

# %% [markdown]
# ## Model 4 : Choice Kernel

# %%
for _ in range(simulation_count):
    alpha_c = 0.1
    beta_c = 3
    actions, rewards = simulate_M4ChoiceKernel(
        trialcount, bandit, alpha_c, beta_c)
    all_sims[4]['actions'].append(actions)
    all_sims[4]['rewards'].append(rewards)

# %% [markdown]
# ## Model 5 : Rescorla-Wagner + Choice Kernel

# %%
for _ in range(simulation_count):
    alpha = 0.1
    beta = 5
    alpha_c = 0.1
    beta_c = 1
    actions, rewards = simulate_M5RWCK(
        trialcount, bandit, alpha, beta, alpha_c, beta_c)
    all_sims[5]['actions'].append(actions)
    all_sims[5]['rewards'].append(rewards)

# %% markdown
# ## WSLS analysis (paper figure A)
# This bit can be a bit confusing. The strategy/algorithm is called win-stay lose-shift, but what we will calculate below is actually win-stay and lose-*stay* values. That's because we then plot the probability of staying depending on previous reward. Matlab code doesn't comment on this, or that the function is called `wsls` but actually returns the values in `lsws` order.

# More to the point; what we will do is plot the probability of making the same choice as on the previous trial, $p(stay)$ on the y-axis, against whether a reward was received on the last trial or not (0 or 1 for no reward or reward, respectively) on the x-axis. This gives us some idea about the difference in behaviour between the models and allows us to consider if what we see makes sense. For example, the win-stay, lose-shift model (Model 2) should always stay ($p(stay) = 1$) if the last action was rewarded (`previous reward = 1`) and always switch ($p(stay) = 0$) if the last action was not rewarded (`previous reward = 0`).

# ### create dataframe

# By creating a dataframe and organizing the data with the principles of "tidy data", we can plot with fewer lines of code and also get the bonus of automatic variance shadowing for each line. An additional bonus is that we can easily try other types of plots; because seaborn understands the data format, we don't have to format the data for each specific plot.

# [Tidy data](http://www.jstatsoft.org/v59/i10/paper) is not what the paper is about, but definitely worth the time when you're exploring your data and may not be sure what type of plot is best to make you understand what's going on. The tricky part in my experience is saving the simulation data in a way so you can separate running the simulations and creating the dataframe. We want to separate those two, as creating and manipulating the dataframe is fairly computationally expensive.

# %%
columns = ['p(stay)', 'previous reward', 'model']
winstay_dataframe = pd.DataFrame(columns = columns)

for model in range(1, len(all_sims)):  # model0 is our dummy entry

    rows = []
    for participant in range(simulation_count):
        ws, ls = winstay_losestay(
            all_sims[model]['actions'][participant],
            all_sims[model]['rewards'][participant]
        )
        rows.append((ws, 1, f"M{model}"))
        rows.append((ls, 0, f"M{model}"))

    model_df = pd.DataFrame(columns = columns, data = rows)
    winstay_dataframe = pd.concat([winstay_dataframe, model_df])

# %% [markdown]

# ### plot figure a
# %%
# setup the look of our plots
sns.set(rc = {  # rc sends options to matplotlib, which seaborn is based on
    "figure.figsize": (5, 6),
    "figure.dpi": 100,
    "lines.markersize": 15,
    "lines.linewidth": 3
})
sns.set_style("darkgrid")
sns.set_palette("colorblind")

fig = sns.lineplot(
    data = winstay_dataframe,
    x = 'previous reward',
    y = 'p(stay)',
    hue = 'model',
    marker = 'o'
)
# we can use ; to suppress showing ticks and limits in text
fig.set(ylim = (0, 1), yticks = (0, 0.5, 1), xticks = (0, 1));
# fig.legend(['M1: random', 'M2: WSLS', 'M3: RW', 'M4: CK', 'M5: RW+CK'])

# %% markdown

# This is where we stop and smell the figure, so to speak. Does the behaviour of our models make sense?

# If any of the models would have performance that varied wildly between participants (simulation runs), we would see that here. Thankfully, it looks like M4 is the only model where $p(stay)$ has visible variance, and even there it's not much. So I think we're good!

# This is with 1000 trials, i.e. 1000 pulls. Would that variance be higher with 100 pulls? We can easily check that by changing `trialcount` in the experimental parameters above and rerun the simulations. Same goes for `simulation_count`; how is the behaviour affected by fewer participants?

# %% markdown

# ## p(correct) analysis (paper figure B)

# Now we're going to look at a single model, Model 3, and how it behaves for different combinations of its two parameters $\alpha$ and $\beta$.

# We create arrays for the parameter values to test, and set the number of simulations (participants) for each parameter combination to `100`.

# The simulation data is saved in a dataframe with columns for each parameter value and the proportions of correct choices overall, for the first 10 trials and for the last 10 trials. Total number of trials is the same as what's set at the top of this file; `trialcount`.

# Each row of this dataframe is one single simulation, so we are again using the tidy data format, which allows seaborn to automagically plot variance intervals for us.

# On my laptop I notice a definite delay before the plot appears if I've set `sim_count` to 1000 instead of 100. So that's the downside of being able to plot "dynamically". Heavy work to calculate all those means and variances :)

# Small note: the paper says the following two figures uses 100 simulations per parameter combination, but I suspect it's actually 1000, based on how the lineplots look and that the matlab code uses 1000. The general pattern is the same though so it doesn't matter that much.

# %%
alphas = np.arange(0.02, 1.02, 0.02)  # upper limit 1.02 is not inclusive
betas = np.array([1, 2, 5, 10, 20])
sim_count = np.arange(100)

rows = []
columns = [
    'alpha', 'beta', 'simcount', 'correct', 'correct early', 'correct late'
]

# product creates a generator for all combinations of array contents
for alpha, beta, simnum in product(alphas, betas, sim_count):

    actions, rewards = simulate_M3RescorlaWagner(
        trialcount, bandit, alpha, beta)

    imax = np.argmax(bandit)  # index of the best arm
    rows.append((
        alpha,
        beta,
        simnum,
        np.mean(actions == imax),  # overall correctness
        np.mean(actions[:10] == imax),  # first 10 trials
        np.mean(actions[-10:] == imax)  # last 10 trials
    ))

all_corrects = pd.DataFrame(columns = columns, data = rows)

# %% markdown

# ## plot early trials

# %%
fig2 = sns.lineplot(
    data = all_corrects,
    x = 'alpha',
    y = 'correct early',
    hue = 'beta',
    palette = 'colorblind'
)
fig2.set(ylim = (0.45, 1), xticks = (0, 0.5, 1));

# %% markdown

# ## plot late trials

# %%
fig3 = sns.lineplot(
    data = all_corrects,
    x = 'alpha',
    y = 'correct late',
    hue = 'beta',
    palette = 'colorblind'
)
fig3.set(ylim = (0.5, 1.05), xticks = (0, 0.5, 1));

# %% markdown

# ## plot overall correctness

# Let's also plot the overall correctness across all trials for each of the parameter combinations, since we have that saved in our dataframe anyway.

# %%
fig4 = sns.lineplot(
    data = all_corrects,
    x = 'alpha',
    y = 'correct',
    hue = 'beta',
    palette = 'colorblind'
)
fig4.set(ylim = (0.5, 1), xticks = (0, 0.5, 1));

# %% markdown

# Do the results make sense? Remember, the model has internal values for each arm.

# The $\alpha$ parameter - learning rate - adjusts how much the internal value will change based on the received reward. Big $\alpha$, big changes (or steps as it's sometimes called). Small $\alpha$, small changes.

# The $\beta$ parameter - inverse temperature - regulates how to pick what arm to pull next. Large $\beta$ values are "greedy", meaning the arm with the larger internal value will more often be picked. With low $\beta$ values, there's more exploration or in other words; more random behaviour.
