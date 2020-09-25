"""
    so what we are doing now is Figure3B_localminima.m
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd

from Figure3_functions import simulate  # since we are in main/parent folder we dont need . before Figure3_functions

# autoreload files in ipython
%load_ext autoreload
%autoreload 2
# %%

realrho = 0.9
realK = 4
realbeta = 10
realalpha = 0.1

# %timeit simdata = simulate(realrho, realK, realbeta, realalpha)

simdata = simulate(realrho, realK, realbeta, realalpha)

# FITTING PARAMETERS

# alphas = np.arange(0.05, 1.05, 0.05)  # i have to say matlab syntax is nicer here as it includes the upper number so you don't have to make it look this weird
# betas = np.append([1], np.arange(5, 55, 5))
# rhos = np.arange(0, 1.05, 0.05)
# Ks = np.arange(2,7)

# testing parameter ranges (above gives 23100 combinations, below only 3000)
# so if 23100 combos took 16 minutes previously, it should now take around 2.
# took 2min 44 secs
# okay now 375 combos, should take only like 20 secs
# yep, 17.8 secs, this'll be good to start testing improvements

alphas = np.arange(0.05, 1, 0.2)
betas = np.arange(1, 30, 10)
rhos = np.arange(0.05, 1, 0.2)
Ks = np.arange(2, 7)
# %% [markdown]

# so matlabs fmincon find the minimum of a function, which is why they return the -llh in their function they feed to fmincon
# in python we have scipy.optimize.minimize which is similar in that it finds the minimum of a function. so all we have to do is to modify our current likelihood function to return the negative sum insted of what it does currently.
# perhaps we also need to modify it to not have a dataframe to manipulate but thats a later step first lets see if it can find a minimum like our brute force method.

# %%

# so what we want to do is
# minimize(recover_participant, [startparamvalues/guesstimate])
# potentially add bounds but lets see first what happens

# simdata has our simulated participant trial data
block_numbers = np.unique(simdata['block'])
blocks = [simdata[simdata['block'] == b] for b in block_numbers]

# so here we give an initial guess of the tuple (rho, K, beta, alpha) and the argument blocks which is our trial data
minimize(recover_participant, (0.8, 3, 15, 0.2), args=blocks)
# yeah this quickly gives us many errors and spits back estimates of rho being in the thousands, so we need bounds, which is a "sequence of (min, max) tuples", i.e. the first is the bounds of rho, second bounds of K etc
bounds = [(0.01, 1), (2, 6), (1, 25), (0.01, 1)]
minimize(recover_participant, [(0.8, 3, 15, 0.2)], args=blocks, bounds=bounds)

# what this gives back now is a bunch of stuff, most importantly to look for is the "fun" which is , haha, fun, but also the log likelihood value for the answer found, i.e. the same kind of value as our brute force gave us earlier. we also want to check that "success" is True, if it's not we should check "message" to see what might be up. in case of overflow, division by null and similar errors the message will give an indication of what can be done to receive a useful answer.
# what we are mainly interested in here is of course the "x" which is the estimated parameter values. they're not great in this case.

# let's see what happens if we provide the 375 different starting values , and i guess let's see how long that'll take ...
results = []
for startparams in [(0.5, 4, 10, 0.5), (0.8, 3, 15, 0.2), (0.1, 2, 5, 0.8)]:
    result = minimize(recover_participant, startparams, args=blocks, bounds=bounds)
    results.append(result)

results







# %%
# compare to brute force
from itertools import product

# 60 combos in total , timeit gave 1.86s +- 12.4ms
# alphas = np.arange(0.05, 1, 0.5)
# betas = np.arange(1, 30, 10)
# rhos = np.arange(0.05, 1, 0.5)
# Ks = np.arange(2, 7)
# 375 combos in total, 12.8s +- 963ms
alphas = np.arange(0.05, 1, 0.2)
betas = np.arange(1, 30, 10)
rhos = np.arange(0.05, 1, 0.2)
Ks = np.arange(2, 7)

%%timeit
loglikes = []
parsets = []
for params in product(rhos, Ks, betas, alphas):  # note order on parameters
    parsets.append(params)
    loglikes.append(recover_participant(params, blocks))

len(list(product(rhos, Ks, betas, alphas)))

# %% [markdown]

# okay so i'm not sure why they do the fmincon fitting in the first section (and maybe i should comment there instead) , maybe it's just to show how it's done without the additional loops later?
# also in this fmincon they do 10 random starting points, but the random values they generate are all between 0 and 1, which doesn't make sense for K or beta. then again mayhaps they think it doesn't matter what the range of the paramter is, you can just do random between 0 and 1 and it's fine? maybe my bayesian leanings come in here, as i see it as self evident to spread the random generated starting points throughout the likely parameter spectra
# this biggest weirdness however is that they use realK as input to fmincon and for the bruteforce, so okay, THATS the difference between this file and the others, and there's no mention of this in the paper. so it's very unclear what code has been used to create what figure in the paper. we should assume the Figure3_localminima.m is for left picture and Figure3B_localminima.m is for the right picture, buut who knows.
# anyway, first they do fmincon, then a brute force search and then they plot the brute force results. just liek in the previous file. so it's a bit of repetition here, don't really need to do this. (it's basically just to get the heatmap graph in the same figure later i believe)
# and maybe i already mentioned this somewhere but why only plot rho and alpha? like, either i'm missing something obvious or the authors have just forgotten we should check the other parameters. maybe this will all be explained better as we move along the paper
# because then they do the actual schtuff they want to introduce here, namely check iterations before we don't get much of a change anymore, so that's where we shall start now

# %%

results = []
for iteration in range(10):
    startparams = (np.random.rand(), realK, np.random.rand() * 20, np.random.rand())
    result = minimize(recover_participant, startparams, args=blocks, bounds=bounds)
    results.append(result)

results

# %%
# START HERE in matlab file; ITERATE SIMULATION AND FITTING
# okay so below with startpointcount and simcount at 10, takes 11 mins , that's waaaay more than matlab, so should figure out why.
# hmm could be all the blocknumbers calculations etc .. matlab takes 20-30 seconds to do the same thing .. 

startpoint_count = 10
bounds = [(0.01, 1), (2, 6), (1, 25), (0.01, 1)]
simulation_count = 100

simulations = []
when = np.zeros([simulation_count, 3], dtype=int)

for simulation in range(simulation_count):
    simdata = simulate(realrho, realK, realbeta, realalpha)
    block_numbers = np.unique(simdata['block'])
    blocks = [simdata[simdata['block'] == b] for b in block_numbers]

    # each row of results will be [rho, K, alpha, beta, loglikelihood] like in matlab code
    results = np.zeros([startpoint_count, 5])

    for startpoint in range(startpoint_count):
        # rho, K, beta, alpha
        startparams = (np.random.rand(), realK, np.random.rand() * 20, np.random.rand())
        result = minimize(recover_participant, startparams, args=blocks, bounds=bounds)

        results[startpoint] = [result.x[0], result.x[1], result.x[2],
        result.x[3], result.fun]

    # at what startpoint did we get global (across all startpoints) minimum?
    global_mindex = np.argmin(results[:, -1])  # i like the word 'mindex' :P
    when[simulation][0] = global_mindex

    # at what startpoint did we get within 0.01 of the global min startpoint?
    closest = np.where(results[:, -1] <= results[:, -1][global_mindex] + 0.01)
    when[simulation][1] = closest[0][0]  # where returns a tuple ([indeces], )

    # at what startpoint did we get within 0.1 of the global min startpoint?
    close = np.where(results[:, -1] <= results[:, -1][global_mindex] + 0.1)
    when[simulation][2] = close[0][0]

    simulations.append(results)

# %%
# to recreate the big figure b in paper, distance between parameters
from scipy.spatial.distance import euclidean

distances_per_simulation = []

for simindex, simulation in enumerate(simulations):
    # get all parameter sets for simulation with index 1
    # params = simulations[1]

    # for second sim (index 1) we reached global max at iteration when[1][0], so we want to compare the parameter values we got there with all the others so lets g
    maxindex = when[simindex][0]
    maxparams = simulation[maxindex][:-1]  # we don't want the loglikelihood which is the last number on each row

    # calculate distances between global max parameters and each estimated parameter set
    # distances = [euclidean(maxparams, paramset[:-1]) for paramset in params]
    distances = [euclidean(maxparams, paramset[:-1]) for paramset in simulation]
    distances_per_simulation.append(distances)



# haha i've done all this backwards, this is horrible, i try to follow the matlab code and then at the end realise we need data in nice format for seaborn
iterations = []
simulations = []
distances = []
for simnum, iterationlist in enumerate(distances_per_simulation):
    simulations.append(np.repeat(simnum, len(iterationlist)))
    for iterationnum, distance in enumerate(iterationlist):
        iterations.append(iterationnum)
        distances.append(distance)


data = {'iteration': iterations, 'simulation': np.array(simulations).flatten(), 'distance': distances}

df = pd.DataFrame(data)


sns.set()
fig = sns.lineplot(data=df, x='iteration', y='distance')
fig.set(xticks=np.arange(10, dtype=int));

df



# %%

# not sure i actually care so much about about their figure 3B to be honest. sure, the main point there is that one should use several random starting points in order to find the global optimum. but first of all it's weird they don't try to recover 1 of 4 parameters (K).
# second of all; a more interesting and much more important thing to show is that depending on the simulation, you get different distances from the parameters you simulated with! again maybe they show this later.. actually yes they show this in the next secion of the paper! i guess we should just try and move on from this thing it's taken too much time at this point.





















# %%
from Figure3_functions import likelihood_RLWM
def recover_participant(parameters, blocks):
    rho = parameters[0]
    K = parameters[1]
    beta = parameters[2]
    alpha = parameters[3]

    loglike_allblocks = 0

    for block in blocks:
        loglike_allblocks += likelihood_RLWM(rho, K, beta, alpha, block)

    return -loglike_allblocks

#%%





































# %%
# FITTING PLAY
from itertools import product
from Figure3_functions import likelihood_RLWM
# %%
%%timeit
def profile_test():
    loglikes = []
    combos = product(alphas, betas, rhos, Ks)
    df = simdata

    block_numbers = np.unique(df['block'])
    blocks = [df[df['block'] == b] for b in block_numbers]  # in our case, memory use is less important than computational speed so this should be fine
    # ooh, this decreased from 22s to 16s. this line takes 10000*1e-06s so the below loglike_allblocks line now takes 2835*1e-06 each hit, down from 3899, and that line is hit 5625 times so .. yeah.. :D
    # so basically, 99.9% of the time taken for this function is now the likelihood_rlwm line!

    for alpha, beta, rho, K in product(alphas, betas, rhos, Ks):
        loglike_allblocks = 0
        # for block in np.unique(df['block']):
        for block in blocks:
            # print(block)
            # loglike_allblocks += likelihood_RLWM(rho, K, beta, alpha, df[df['block'] == block])
            loglike_allblocks += likelihood_RLWM(rho, K, beta, alpha, block)
        loglikes.append([alpha, beta, rho, K, loglike_allblocks])

    fit_results = pd.DataFrame(columns=["alpha", "beta", "rho", "K", "loglike"], data=loglikes)

# %%
%load_ext line_profiler
%lprun -f likelihood_RLWM likelihood_RLWM(rho, K, beta, alpha, df[df['block'] == 3])
%lprun?

# %%
# cant get this to output in hydrogen even with below redirect oh well have to use ipython terminal instead
def page_printer(data, start=0, screen_lines=0, pager_cmd=None):
    if isinstance(data, dict):
        data = data['text/plain']
    print(data)
from IPython.core import page
page.page = page_printer
# %%
%lprun -f simulate simulate(realrho, realK, realbeta, realalpha)




# %%
with perf_timer('blabla') as pf:
    simulate(realrho, realK, realbeta, realalpha)

# %%
import time
class perf_timer:
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if self.name:
            print('%s: elapsed time %.3f'%(self.name, self.elapsed))
