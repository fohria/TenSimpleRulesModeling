# so, random thought, do they mention simulating with a specific parameter value set to check the spread of performance? hmm, quick check of figure 2/box2 is they kind of do that, but without error bars, i think i have a note actually, somewhere, suggesting we can add that!

import numpy as np
import seaborn as sns

%load_ext autoreload
%autoreload 2
# %%

# so what box4 says is we have the same task as in box2/fig2, namely the two armed bandit task. one arm has probability 0.2 for reward and the other p=0.8 to receive reward. and we do 1000 trials. this is, of course, a best case scenario type, in a real experiment it would be very unlikely we would get a person to do 1000 trials of this. the algorithm may be able to learn way more quickly than 1000 trials what arm is best, but a good question here is how many trials are needed to be able to recover the parameters with some certainty? let's come back to that question later.

# so, next, to simulate one (ro)bot participant, we use model3, Rescorla-Wagner. this model has two parameters; learning rate alpha and softmax temperature beta. the authors suggest picking random parameter values so let's visualise those values to get a feel for what values to expect.

# %%

sns.set()  # to reset look of graphs, gives a decent checkboard background

# alpha ~ U(0, 1)
# this one may not really need a visualisation if you're familiar with probability distributions but i like visualisations and maybe you're not familiar with distributions. so U is "uniform" and just means all values between, in this case, 0 and 1 are equally likely:

sns.distplot(np.random.uniform(0, 1, 10000))  # the last number is how many numbers to draw

# beta ~ Exp(10)
# an exponential distribution may be more difficult to visualise , especially when it's modified by its "rate parameter", in this case, 10

sns.distplot(np.random.exponential(10, 10000))
# we can see in the graph that we can expect most values to be below 10, definitely below 20.
# briefly, what this means is that if our model currently has the "values" for each arm as [0.4, 0.5], the bigger beta is, the more likely the bigger value will be more probably. i.e. bigger beta, more greedy; will always pick the higher valued option no matter how small the difference between the values. the closer to zero, the more random the behaviour gets and even a small difference between the option values will mean nothing.
beta = 10
np.exp(beta * np.array([0.4, 0.5])) / sum(np.exp(beta * np.array([0.4, 0.5])))

# %%

from SimulationFunctions.simulate_M3RescorlaWagner_v1 import simulate_M3RescorlaWagner_v1 as simulate
from LikelihoodFunctions.lik_M3RescorlaWagner_v1 import lik_M3RescorlaWagner_v1 as likelihood
from scipy.optimize import minimize

# simulate and fit 1000 times
sim_and_fit_count = 1000
# what box 4 now says is that we do a simulation, then try to fit parameters based on the observed choices. just like we would with human participants. each participant does 1000 trials and the bandit arms have 0.2 and 0.8 probabilities:

trial_count = 1000  # T in paper's terms
bandit = [0.2, 0.8]  # mu in paper's terms

sims_and_fits = []
# we should define bounds for the fitting function to avoid crashes
bounds = [(0.01, 1), (1, 60)]  # as we saw above, beta=60 is very unlikely but possible
starting_guess = (0.5, 1 / 10)  # we use the means of each distribution as our starting guess, mean of exponential function is 1 / lambda

for counter in range(sim_and_fit_count):
    alpha = np.random.uniform(0, 1)
    beta = np.random.exponential(10)
    actions, rewards = simulate(trial_count, bandit, alpha, beta)

    fitresult = minimize(likelihood, starting_guess,
                         args=(actions, rewards), bounds=bounds)
    fitalpha = fitresult.x[0]
    fitbeta = fitresult.x[1]
    sims_and_fits.append([alpha, beta, fitalpha, fitbeta])

# %%
# let's plot!
import pandas as pd
# %%

column_names = ['realalpha', 'realbeta', 'fitalpha', 'fitbeta']
data = pd.DataFrame(columns=column_names, data=sims_and_fits)
data.to_csv('figure4.csv', index=False)

# find bad apples. i mean alphas.
data.loc[:, 'badalpha'] = np.abs(alphas['realalpha'] - alphas['fitalpha']) > 0.25
# seaborn wont like trying to plot trues and falses
data.loc[:, 'badalpha'] = data['badalpha'].apply(lambda x: 'bad alpha' if x else 'good alpha')

data.badalpha.map({'bad alpha': 666, 'good alpha': 999})

# alphas['badalpha'] = np.abs(alphas['realalpha'] - alphas['fitalpha']) > 0.25
# the above line will most likely give you a "SettingWithCopyWarning", which in this case can be disregarded. they would like you to do the assignment as: alphas.loc[:, 'badalpha'] = blabla
# in my opinion pandas has this problem where they prefer defaulting to complicated syntax to do fairly straighforward things. i don't think this warning should be shown when you are obviously creating a new column but what do i know? *shrug*
# after reading the pandas documentation linked in the warning message i understand the issue better, but this stackoverflow anser says it better: "explicit is better than implicit": https://stackoverflow.com/questions/38886080/python-pandas-series-why-use-loc
# that answer also shows some unpredictable things, like how df[1:2] gives you a different result than df.loc[1:2]
# like, i _get_ that there are issues here that may be inherent to python i just feel like the pandas interface by default is kinda obtuse; why not default to df[:, 'columnname'] instead of using loc?
# i'm familiar with R and can use it, but i've more experience with python. R has tools that are much more intuitive though, so i can really get comments like [insert twitter quote about losing six years of your life]. anyway, i digress..
# by the way this would give the same warning: alphas['badalpha'] = alphas.apply(lambda x: np.abs(x[0] - x[1]) > 0.25, axis=1)

# bug in matplotlib 3.3.1 so hue doesnt work
# workaround https://stackoverflow.com/questions/63443583/seaborn-valueerror-zero-size-array-to-reduction-operation-minimum-which-has-no

sns.scatterplot(data=data, x='realalpha', y='fitalpha', hue=data.badalpha.tolist())

# notice we here mark bad alphas, not bad betas
sns.scatterplot(data=data, x='realbeta', y='fitbeta', hue=data.badalpha.tolist())

# as in the paper we see that fits are farily good < 10 and then get increasingly worse
# hmm but it also looks like there are many more bad alphas in the alphas plot than in the betas plot, so lets plot only the betas that are badalphas so to speak
sns.scatterplot(data=data.query('badalpha == "bad alpha"'), x='realbeta', y='fitbeta')

# ahh it looks like all of them are likely around 0;
fig = sns.scatterplot(data=data.query('badalpha == "bad alpha"'), x='realbeta', y='fitbeta')
fig.set(xlim=(0,2), ylim=(0,5))
# and here we see it's when realbeta is < 1 that the fitting has issues. and this makes sense, because with a very low beta, the behaviour becomes basically random. any model would have issues fitting well if there is no pattern in the behaviour.

# comment after the graph: general patterns are one thing, but how certain can we be that a single case has been recovered well? the recovery is made on ondividual level, meaning we fit one pair of parameters for one participant, another pair of parameters for the next participant and so on. in each case, how do we know if it's a "bad" recovery or a "good" one? the simple answer is: we don't. we can only get a general idea of how far, on average, we are from the true parameters. let's see if they bring this issue up.
# TODO: calculate distance of each recovered parameter pair from the true values, and graph that distance with uncertainty. or do it separately for each parameter.

# ahh, we calculate abs distance for alpha and beta and then plot those
data.loc[:, 'alphadistance'] = np.abs(data.realalpha - data.fitalpha)
data.loc[:, 'betadistance'] = np.abs(data.realbeta - data.fitbeta)
# could actually do regular distance
data.loc[:, 'alphadistance'] = data.realalpha - data.fitalpha
data.loc[:, 'betadistance'] = data.realbeta - data.fitbeta

alphastats = data.alphadistance.describe(percentiles=[0.05, 0.95])
from scipy import stats
# since each simulation and fit are separate, we treat them as single draws
ci_low, ci_up = stats.norm.interval(0.95, loc=alphastats['mean'], scale=alphastats['std'])

fig = sns.distplot(data.alphadistance)
fig.set(xlim=(-0.5, 0.5))
fig.axvline(ci_low)
fig.axvline(ci_up)

# so, in short, we can expect that learning rate alpha will be off by 0.3 in 95% of our cases. is that acceptable? i can't say, i guess that depends on the experimental task. we could go back and simulate to see how much behaviour/performance is impacted on the task when we vary alpha 0.3 for example.

betastats = data.betadistance.describe()
ci_low, ci_up = stats.norm.interval(0.95, loc=betastats['mean'], scale=betastats['std'])

fig = sns.distplot(data.betadistance)
# fig.set(xlim=(-0.4, 0.4))
fig.axvline(ci_low)
fig.axvline(ci_up)

# to more easily compare to alpha, we normalize beta to 0-1
# x = x - xmin / xmax - xmin
# xmax = betastats['max']
# xmin = betastats['min']
# beta_normalized = data.betadistance.apply(lambda x: (x - xmin) / (xmax-xmin))
# x = (x - mean) / std
beta_normed = data.betadistance.apply(lambda x: (x - betastats['mean']) / betastats['std'])
betanorm_stats = beta_normed.describe()

ci_low, ci_up = stats.norm.interval(0.95, loc=betanorm_stats['mean'], scale=betanorm_stats['std'])

fig = sns.distplot(beta_normed)
# fig.set(xlim=(-0.4, 0.4))
fig.axvline(ci_low)
fig.axvline(ci_up)

# okay maybe scaling wasnt that useful . important point is:
# so what we see here, unfortunately, is that recovery for beta is very uncertain. if we have one participant, and we fit this model, we can have a 95% confidence that the recovered beta parameter
