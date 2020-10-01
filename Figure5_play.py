# from paper "As with parameter recovery, we believe that the best approach is to match the range of the parameters to the range seen in your data, or to the range that you expect from prior work."
# so, i know what we're doing is fiddly and messy, but is the advice to look at your experimental data really the best? it feels like there's an issue methodologically there, unless the data you're talking about is explicitly a pilot experiment to explore what parameter ranges your human participant data gives you.

"""
    BOX/FIGURE 5 : confusion matrix and inversion matrix

    TODO:
    - [DONE] change all simulation functions to return numpy arrays
    - [DONE] change all simulation functions to compile with numba
    - [DONE] change all likelihood functions to compile with numba
    - add fitsuccess check to all fitting functions
    - benchmark change in creating the confusion matrix!
"""

# %%

# testing numba likelihood methods

from LikelihoodFunctions.lik_M1random_v1 import lik_M1random_v1

start_guess = [np.random.uniform(0, 1)]
bounds = [(0.01, 0.99)]

fit_success = False
max_iterations = 10
while fit_success is False and max_iterations != 0:
    fitresult = minimize(lik_M1random_v1,
                         start_guess,
                         args=(actions, rewards),
                         bounds=bounds)
    fit_success = fitresult.success
    max_iterations -= 1

from LikelihoodFunctions.lik_M2WSLS_v1 import lik_M2WSLS_v1

start_guess = [np.random.uniform(0, 1)]
bounds = [(0.01, 0.99)]

fit_success = False
max_iterations = 10
while fit_success is False and max_iterations != 0:
    fitresult = minimize(lik_M2WSLS_v1,
                         start_guess,
                         args=(actions, rewards),
                         bounds=bounds)
    fit_success = fitresult.success
    max_iterations -= 1

from LikelihoodFunctions.lik_M3RescorlaWagner_v1 import lik_M3RescorlaWagner_v1

start_guess = [np.random.uniform(0, 1), np.random.exponential(1)]
bounds = [(1e-10, 1 - 1e-10), (1e-10, 20)]

fit_success = False
max_iterations = 10
while fit_success is False and max_iterations != 0:
    fitresult = minimize(lik_M3RescorlaWagner_v1,
                         start_guess,
                         args=(actions, rewards),
                         bounds=bounds)
    fit_success = fitresult.success
    max_iterations -= 1

from LikelihoodFunctions.lik_M4CK_v1 import lik_M4CK_v1

start_guess = [np.random.uniform(0, 1), np.random.exponential(1)]
bounds = [(1e-10, 1 - 1e-10), (1e-10, 20)]

fit_success = False
max_iterations = 10
while fit_success is False and max_iterations != 0:
    fitresult = minimize(lik_M4CK_v1,
                         start_guess,
                         args=(actions, rewards),
                         bounds=bounds)
    fit_success = fitresult.success
    max_iterations -= 1

from LikelihoodFunctions.lik_M5RWCK_v1 import lik_M5RWCK_v1

start_guess = [np.random.uniform(0, 1), np.random.exponential(1),
               np.random.uniform(0, 1), np.random.exponential(1)]
bounds = [(1e-10, 1 - 1e-10), (1e-10, 20),
          (1e-10, 1 - 1e-10), (1e-10, 20)]

fit_success = False
max_iterations = 10
while fit_success is False and max_iterations != 0:
    fitresult = minimize(lik_M5RWCK_v1,
                         start_guess,
                         args=(actions, rewards),
                         bounds=bounds)
    fit_success = fitresult.success
    max_iterations -= 1

# %%

# autoreload
%load_ext autoreload
%autoreload 2

# "Note that this measure, which we term the ‘inversion matrix’ to distinguish it from the confusion matrix, is not the same as the confusion matrix unless model recovery is perfect."
# --- not sure i understand the "unless model recovery is perfect", but perhaps it will become more clear as we go along

# anyway, now we do, for each model, 100 simulations and for each simulation we fit models 1-5.

# lets start with model3 since we have a likelihood function for that already, so we do row 3 of confusion matrix A in fig/box5. ah wait, we need to fit all models, haha. okay, just start doing likelihood functions for each.
import numpy as np
from SimulationFunctions.simulate_M1random_v1 import simulate_M1random_v1
from SimulationFunctions.simulate_M2WSLS_v1 import simulate_M2WSLS_v1
from SimulationFunctions.simulate_M3RescorlaWagner_v1 import simulate_M3RescorlaWagner_v1
from SimulationFunctions.simulate_M4ChoiceKernel_v1 import simulate_M4ChoiceKernel_v1
from SimulationFunctions.simulate_M5RWCK_v1 import simulate_M5RWCK_v1
from FittingFunctions.fit_all_v1 import fit_all_v1
# from FittingFunctions.fit_M1random_v1 import fit_M1random_v1
# from FittingFunctions.fit_M2WSLS_v1 import fit_M2WSLS_v1
# from FittingFunctions.fit_M3RescorlaWagner_v1 import fit_M3RescorlaWagner_v1
# from FittingFunctions.fit_M4CK_v1 import fit_M4CK_v1
# from FittingFunctions.fit_M5RWCK_v1 import fit_M5RWCK_v1


exp_trialcount = 1000
bandit = np.array([0.2, 0.8])
beta_increase = 0  # set to 0 for figure A, 1 for figure B
simfit_runcount = 10  # how many times to simulate each model and fit

# %%

# with fit_all_v1 function done, we can easily just do entire confusion matrix

confusion_matrix = np.zeros([5, 5])
# confusion_matrix = np.zeros(5)

for simfitrun in range(simfit_runcount):
    print(f"simfitrun {simfitrun}")

    # model 1
    bias = np.random.uniform(0, 1)
    actions, rewards = simulate_M1random_v1(exp_trialcount, bandit,
                                            bias)
    _, _, best = fit_all_v1(actions, rewards)
    confusion_matrix[0, :] += best

    # model 2
    epsilon = np.random.uniform(0, 1)
    actions, rewards = simulate_M2WSLS_v1(exp_trialcount, bandit,
                                          epsilon)
    _, _, best = fit_all_v1(actions, rewards)
    confusion_matrix[1, :] += best

    # model 3
    alpha = np.random.uniform(0, 1)
    beta = beta_increase + np.random.exponential(1)
    actions, rewards = simulate_M3RescorlaWagner_v1(exp_trialcount, bandit,
                                                    alpha, beta)
    _, _, best = fit_all_v1(actions, rewards)
    confusion_matrix[2, :] += best

    # model 4
    alpha_c = np.random.uniform(0, 1)
    beta_c = beta_increase + np.random.exponential(1)
    actions, rewards = simulate_M4ChoiceKernel_v1(exp_trialcount,
                                                  bandit,
                                                  alpha_c, beta_c)
    _, _, best = fit_all_v1(actions, rewards)
    confusion_matrix[3, :] += best

    # model 5
    alpha = np.random.uniform(0, 1)
    beta = beta_increase + np.random.exponential(1)
    alpha_c = np.random.uniform(0, 1)
    beta_c = beta_increase + np.random.exponential(1)
    actions, rewards = simulate_M5RWCK_v1(exp_trialcount,
                                          bandit,
                                          alpha, beta,
                                          alpha_c, beta_c)
    _, _, best = fit_all_v1(actions, rewards)
    confusion_matrix[4, :] += best


sns.heatmap(confusion_matrix / simfit_runcount, annot=True)

# %%


# :::::::::::::: NOTES BELOW HERE ::: (yes, not very tidy :)

# %%
# updating simulation functions to numba
exp_trialcount = 1000
bandit = np.array([0.2, 0.8])
bias = np.random.uniform(0, 1)
actions, rewards = simulate_M1random_v1(exp_trialcount, bandit, bias)
epsilon = np.random.uniform(0, 1)
actions, rewards = simulate_M2WSLS_v1(exp_trialcount, bandit, epsilon)
alpha = np.random.uniform(0, 1)
beta_increase = 1
beta = beta_increase + np.random.exponential(1)
actions, rewards = simulate_M3RescorlaWagner_v1(
    exp_trialcount, bandit, alpha, beta
)  # hah, wonder what pep8 says about this
alpha_c = np.random.uniform(0, 1)
beta_c = beta_increase + np.random.exponential(1)
actions, rewards = simulate_M4ChoiceKernel_v1(
    exp_trialcount, bandit, alpha_c, beta_c)
alpha = np.random.uniform(0, 1)
beta = beta_increase + np.random.exponential(1)
alpha_c = np.random.uniform(0, 1)
beta_c = beta_increase + np.random.exponential(1)
actions, rewards = simulate_M5RWCK_v1(
    exp_trialcount, bandit, alpha, beta, alpha_c, beta_c)

# %%

# TEST NUMBA WEEE

# got OMP Error 15 about llvl and MKL. it may be a macos specific issue: https://stackoverflow.com/a/58869103
# so trying to reinstall conda environment then

exp_trialcount = 1000
bandit = np.array([0.2, 0.8])
bias = np.random.uniform(0, 1)
simfitcount = 1

from scipy.optimize import minimize
from lik_M5_numba import lik_M5_numba


def test_numba():
    for bla in range(simfitcount):
        actions, rewards = simulate_M1random_v1(exp_trialcount,
                                                bandit,
                                                bias)

        # numba wants numpy arrays to infer types
        actions = np.array(actions)
        rewards = np.array(rewards)

        start_guess = [np.random.uniform(0, 1), np.random.exponential(1),
                       np.random.uniform(0, 1), np.random.exponential(1)]
        # to avoid division by 0 we don't want to include 0 and 1 in bounds
        # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
        bounds = [(1e-10, 1 - 1e-10), (1e-10, 20),
                  (1e-10, 1 - 1e-10), (1e-10, 20)]

        fit_success = False
        max_iterations = 10
        while fit_success is False and max_iterations != 0:
            fitresult = minimize(lik_M5_numba,
                                 start_guess,
                                 args=(actions, rewards),
                                 bounds=bounds)
            fit_success = fitresult.success
            max_iterations -= 1
test_numba()
%timeit test_numba()

%timeit test_org()

from LikelihoodFunctions.lik_M5RWCK_v1 import lik_M5RWCK_v1
def test_org():
    for bla in range(simfitcount):
        actions, rewards = simulate_M1random_v1(exp_trialcount,
                                                bandit,
                                                bias)

        start_guess = [np.random.uniform(0, 1), np.random.exponential(1),
                       np.random.uniform(0, 1), np.random.exponential(1)]
        # to avoid division by 0 we don't want to include 0 and 1 in bounds
        # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
        bounds = [(1e-10, 1 - 1e-10), (1e-10, 20),
                  (1e-10, 1 - 1e-10), (1e-10, 20)]

        fit_success = False
        max_iterations = 10
        while fit_success is False and max_iterations != 0:
            fitresult = minimize(lik_M5RWCK_v1,
                                 start_guess,
                                 args=(actions, rewards),
                                 bounds=bounds)
            fit_success = fitresult.success
            max_iterations -= 1

# %%
from numba import jit

for simfitrun in range(1):
    print(f"simfitrun {simfitrun}")
    bias = np.random.uniform(0, 1)
    actions, rewards = simulate_M1random_v1(exp_trialcount,
                                            bandit,
                                            bias)
    rewards =
    for fittingrun in range(10):
        print(f"fittingrun {fittingrun}")
        result = fit_M5RWCK_v1(actions, rewards)


# %%

# %%

# so, one simfit run for model1 takes 14-15s. running line profiler on fit_all_v1 says:
"""
    13         1      24827.0  24827.0      0.2      _, _, BICs[0] = fit_M1random_v1(actions, rewards)
    14         1      47558.0  47558.0      0.3      _, _, BICs[1] = fit_M2WSLS_v1(actions, rewards)
    15         1     377424.0 377424.0      2.6      _, _, BICs[2] = fit_M3RescorlaWagner_v1(actions, rewards)
    16         1    3195522.0 3195522.0     22.3      _, _, BICs[3] = fit_M4CK_v1(actions, rewards)
    17         1   10712083.0 10712083.0     74.6      _, _, BICs[4] = fit_M5RWCK_v1(actions, rewards)
"""

# 22.3% of the time is model4 and 75% on model5. lets check model5 what is taking so much time.

# haha, of course that one says 100% of the time taken is minimize function. we may want to parallelize to increase speed, but first we should probably check what we can do to the function itself, likely (hah!) the likelihood function

# crap, each np.clip takes ~30% of the time so together they account for 2/3 of the total computation time!

# therefore, testing to change to float128 for Q and CK in likem5:
for simfitrun in range(1):
    print(f"simfitrun {simfitrun}")
    bias = np.random.uniform(0, 1)
    actions, rewards = simulate_M1random_v1(exp_trialcount,
                                            bandit,
                                            bias)
    for fittingrun in range(1):
        print(f"fittingrun {fittingrun}")
        result = fit_M5RWCK_v1(actions, rewards)

# actually that seems to have done the trick. have tried above code, i.e. 20 simfitruns and 10 fitrun (11min14secs) for each of them and no underflow errors, even though we removed the clips. changing the other likelihood functions in same fashion

# profiling m5 likelihood again we see that calculating action probabilities takes ~60% of total time, and total time is <0.05s. that's perfectly fine, it means there's nothing obvious to optimize:
"""
18                                                   # compute choice probabilities
    19      1000       5781.0      5.8     12.2          V = beta * Q + beta_c * CK
    20      1000      23411.0     23.4     49.3          p = np.exp(V) / np.sum(np.exp(V))
"""
# it WOULD be nice to optimize it more, because when we again do the fit_all profiling we see m5 is 92.5% of the computational time:
"""
    13         1      22225.0  22225.0      0.4      _, _, BICs[0] = fit_M1random_v1(actions, rewards)
    14         1      40642.0  40642.0      0.7      _, _, BICs[1] = fit_M2WSLS_v1(actions, rewards)
    15         1     182365.0 182365.0      3.3      _, _, BICs[2] = fit_M3RescorlaWagner_v1(actions, rewards)
    16         1     170262.0 170262.0      3.1      _, _, BICs[3] = fit_M4CK_v1(actions, rewards)
    17         1    5129224.0 5129224.0     92.5      _, _, BICs[4] = fit_M5RWCK_v1(actions, rewards)
"""

# lets try this parallelization library i found that simply replaces minimize.
# it installed okay, and seems to be working, not sure it actually improves speed anything though. oooh, actually, i think my laptop usually is treated as single core...yep, when testing on another machine it's now much faster. great!

# %%

BEST = np.zeros(5)

for simfitrun in range(5):
    print(f"simfitrun {simfitrun}")
    BICs = np.zeros(5)
    bias = np.random.uniform(0, 1)
    actions, rewards = simulate_M1random_v1(exp_trialcount,
                                            bandit,
                                            bias)

    # okay so this is one fit_all and then we can just get the 'best', mayhaps they use bic and ibest later
    _, _, BICs[0] = fit_M1random_v1(actions, rewards)
    _, _, BICs[1] = fit_M2WSLS_v1(actions, rewards)
    _, _, BICs[2] = fit_M3RescorlaWagner_v1(actions, rewards)
    _, _, BICs[3] = fit_M4CK_v1(actions, rewards)
    _, _, BICs[4] = fit_M5RWCK_v1(actions, rewards)

    print(BICs)

    mindex = np.argmin(BICs)
    BEST[mindex] += 1

BEST

# %%

# okay, so their confusion matrix actually doesnt use bic values directly, it just counts how many times model X was the best one.
# so for model1:
# simulate_M1random_v1
# best = [1, 0, 0, 0, 0]  # if model1 was best fit
# next round, if model2 was best fit
# best = [0, 1, 0, 0, 0]
# so if we only run 2 rounds, then at the end we just check how many times one model was the better:
# p (fit|sim) -> best / nbrofloops -> best/2 -> [0.5, 0.5, 0, 0, 0]
# hahaha, it's obvious now i see it, but they dont explain it and the paper only mentions BIC so i thought they were using that to create the confusion matrix. i mean they do, just indirectly.

# random comment; i complain about syntax often, but numpy as np, pandas as pd and not to mention seaborn as sns -- those conventions arent the most intuitive either

# %%
# :::::::::::::::: checking range of exp(1)

y = np.random.exponential(1, 100000)
len(y[y > 20]) / 100000
sns.distplot(y)

# %%
# :::::::::::::::: investigation of underflow error

# runtimewarning: underflow encountered in multiply for m4likelihoood:
"""
/TenSimpleRulesModeling/LikelihoodFunctions/lik_M4CK_v1.py:16: RuntimeWarning: underflow encountered in multiply
  p = np.exp(beta_c * CK) / np.sum(exp(beta_c * CK))
/TenSimpleRulesModeling/LikelihoodFunctions/lik_M4CK_v1.py:22: RuntimeWarning: underflow encountered in multiply
  CK = (1 - alpha_c) * CK
"""
# so we use warnings library to find out what exactly happens
# okay so the issue seems to be that sometimes we get this scenario: np.sum(np.exp(1e-10 * np.array([1.e+00, 1.e-300])))
# where 1e-10 * np.array([1.e+00, 1.e-300]) will cause the underflow
# there might be a way of getting around this using sneaky log calculations, and as this https://stackoverflow.com/questions/33434032/avoid-underflow-using-exp-and-minimum-positive-float128-in-numpy says, there's a scipy function logsumexp that may be useful. i can't really figure out how to do that properly, since it's actually about the beta being multiplied with the vector, not really the sum or anything, so for the moment lets try and use numpy.clip instead to just have a very small value close to zero instead of ridiculousely close to zero
beta_c = 1e-10
CK = np.array([1.00000000e+000, 1.00000248e-300])
np.log(np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK)))

np.exp(2)


p = np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK))
np.exp(1e-10 * np.array([1.00000000e+000, 1.00000248e-300]))
1e-10 * np.array([1.00000000e+000, 1.00000248e-300])
np.clip(np.array([1.00000000e+000, 1.00000248e-300]), 1e-50, 1e50)
from scipy.special import logsumexp
np.sum(np.exp(1e-10 * np.array([1.00000000e+000, 1.00000248e-300])))

np.sum(np.exp(1e-10 * np.array([1.e+00, 1.e-50])))

np.exp(logsumexp(np.array([1.00000000e+000, 1.00000248e-300]), b=1e-10))
