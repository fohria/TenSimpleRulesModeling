# so there's a function "simulate_M6RescorlaWagnerBias_v1" but it's never used. at least i can't find that function name anywhere else than in that file, when searching the entire folder structure.

# i'm a bit confused here, the paper makes it sound like we can modify, for example, model3 by adding a bias and thereby capturing some randomness/noise in the simulated data so the two main parameters - alpha and beta - are recovered better. the box6 even says "Example: improving parameter recovery by modeling unimportant parameters."
# but then what they do is simulate a biased model, and show that a model with bias fits better than the model without bias. ... yes? you simulated with bias so how is this interesting? because you're simulating a hypothetical person with such a bias? if so, the bias isn't an "unimportant" parameter, it's actually "there" so the biased model is actually a better model, not "unimportant", no?
# i probably misunderstand again, but wouldn't a more interesting test be to simulate with original M3 and see if the M3+bias helps stabilize recovery of alpha&beta params?

# okay, i'm going to start by doing that because it makes more sense to me and also i'm now curious what will happen..

# %%
%load_ext autoreload
%autoreload 2

import numpy as np
from numba import njit, int32
from scipy.optimize import minimize

from SimulationFunctions.simulate_M3RescorlaWagner_v1 import simulate_M3RescorlaWagner_v1

# %%

# okay, simulating is just M3 10x with mu = [0.2, 0.8] or [0.8, 0.2]
@njit
def simulate():

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
        a_block, r_block = simulate_M3RescorlaWagner_v1(
            trialcount, bandit, alpha, beta)
        actions[block * 50:(block + 1) * 50] = a_block
        rewards[block * 50:(block + 1) * 50] = r_block

    return actions, rewards, [alpha, beta]


@njit
def llh_m3(parameters, actions, rewards):

    alpha = parameters[0]
    beta = parameters[1]

    Q = np.array([0.5, 0.5])

    trialcount = len(actions)
    choice_probs = np.zeros(trialcount)

    for trial in range(trialcount):

        if trial % 50 == 0:  # TODO: make blockcount dynamic?
            Q = np.array([0.5, 0.5])

        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

        choice_probs[trial] = p[actions[trial]]

        delta = rewards[trial] - Q[actions[trial]]
        Q[actions[trial]] += alpha * delta

    loglikelihood = np.sum(np.log(choice_probs))
    return -loglikelihood


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
        p2 = 1/(1+np.exp(beta*(bias+Q[0]-Q[1])))
        probs = np.array([1-p2, p2])

        choice_probs[trial] = probs[actions[trial]]

        delta = rewards[trial] - Q[actions[trial]]
        Q[actions[trial]] += alpha * delta

    loglikelihood = np.sum(np.log(choice_probs))
    return -loglikelihood


def fitm3(actions, rewards):

    # 20 random starting points
    best_fit = 9999
    best_params = []
    for startpoint in range(20):
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
    for startpoint in range(20):
        guess = np.random.rand(3)
        bounds = [(0.01, 0.99), (0.1, 10), (0.01, 0.99)]  # alpha, beta, bias
        result = minimize(
            llh_m3bias, guess, args=(actions, rewards), bounds=bounds)
        if result.fun < best_fit and result.success is True:
            best_fit = result.fun
            best_params = result.x

    return best_fit, best_params


# %%

simfitcount = 100

m3 = []
m3b = []
realparams = []

for simfit in range(simfitcount):

    print(f"simfitrun {simfit}")

    a, r, params = simulate()

    m3.append(fitm3(a, r))
    m3b.append(fitm3bias(a, r))
    realparams.append(params)

m3
m3b
realparams

# %%

import seaborn as sns

m3_alphas = [x[1][0] for x in m3]
m3_betas = [x[1][1] for x in m3]
m3b_alphas = [x[1][0] for x in m3b]
m3b_betas = [x[1][1] for x in m3b]
real_alphas = [x[0] for x in realparams]
real_betas = [x[1] for x in realparams]


fig = sns.scatterplot(x=m3_alphas, y=real_alphas)
fig.set(title='m3 alphas', xlabel='fit alpha', ylabel='real alpha');

fig = sns.scatterplot(x=m3_betas, y=real_betas)
fig.set(title='m3 betas', xlabel='fit beta', ylabel='real beta');

fig = sns.scatterplot(x=m3b_alphas, y=real_alphas)
fig.set(title='m3bias alphas', xlabel='fit alpha', ylabel='real alpha');

fig = sns.scatterplot(x=m3b_betas, y=real_betas)
fig.set(title='m3bias betas', xlabel='fit beta', ylabel='real beta');

m3b_bias = [x[1][2] for x in m3b]
np.mean(m3b_bias)


sns.histplot(m3b_bias)

# so bias is often close to 0 , that makes sense

# is there a huge difference between alpha and beta values between the two models?

alpha_distances = np.array(m3_alphas) - np.array(m3b_alphas)

sns.histplot(alpha_distances)

beta_distances = np.array(m3_betas) - np.array(m3b_betas)

sns.histplot(beta_distances)

# fairly close it seems

# and compared to real alphas/betas?

m3_alpha_distances = np.array(m3_alphas) - np.array(real_alphas)
m3b_alpha_distances = np.array(m3b_alphas) - np.array(real_alphas)

np.mean(m3_alpha_distances)
np.mean(m3b_alpha_distances)
np.std(m3_alpha_distances)
np.std(m3b_alpha_distances)

# so yeah, none of this is surprising i guess, they are the same model after all, it's just the bias one _can_ account for there being a bias. haha. i could've seen that from the start but better late than never i guess!


# - sim M3
# - fit M3 :: need new likelihood function because they changed the task
# - create M3+bias
# - fit M3+bias
