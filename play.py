from SimulationFunctions.simulate_M1random_v1 import simulate_M1random_v1
from SimulationFunctions.simulate_M2WSLS_v1 import simulate_M2WSLS_v1
from SimulationFunctions.simulate_M3RescorlaWagner_v1 import simulate_M3RescorlaWagner_v1
from SimulationFunctions.simulate_M4ChoiceKernel_v1 import simulate_M4ChoiceKernel_v1
from SimulationFunctions.simulate_M5RWCK_v1 import simulate_M5RWCK_v1


import numpy as np
from FittingFunctions.fit_M1random_v1 import fit_M1random_v1
from FittingFunctions.fit_M2WSLS_v1 import fit_M2WSLS_v1
from FittingFunctions.fit_M3RescorlaWagner_v1 import fit_M3RescorlaWagner_v1
from FittingFunctions.fit_M4CK_v1 import fit_M4CK_v1
from FittingFunctions.fit_M5RWCK_v1 import fit_M5RWCK_v1

%load_ext autoreload
%autoreload 2

#%%
exp_trialcount = 1000
bandit = np.array([0.2, 0.8])

for test in range(100):

    print(f"test: {test}")

    actions, rewards = simulate_M1random_v1(
        exp_trialcount, bandit, np.random.uniform(0, 1))
    bla = fit_M5RWCK_v1(actions, rewards)

    actions, rewards = simulate_M2WSLS_v1(
        exp_trialcount, bandit, np.random.uniform(0, 1)
    )
    bla = fit_M5RWCK_v1(actions, rewards)

    actions, rewards = simulate_M3RescorlaWagner_v1(
        exp_trialcount, bandit,
        np.random.uniform(0, 1), 1 + np.random.exponential(1))
    bla = fit_M5RWCK_v1(actions, rewards)

    actions, rewards = simulate_M4ChoiceKernel_v1(
        exp_trialcount, bandit, np.random.uniform(0, 1), 1+np.random.exponential(1)
    )
    bla = fit_M5RWCK_v1(actions, rewards)

    actions, rewards = simulate_M5RWCK_v1(
        exp_trialcount, bandit, np.random.uniform(0, 1), 1+np.random.exponential(1), np.random.uniform(0, 1), 1+np.random.exponential(1)
    )
    bla = fit_M5RWCK_v1(actions, rewards)



# %%
def fit_all(actions, rewards):

    BICs = np.zeros(5)
    loglike = np.zeros(5)

    _, loglike[0], BICs[0] = fit_M1random_v1(actions, rewards)
    _, loglike[1], BICs[1] = fit_M2WSLS_v1(actions, rewards)
    _, loglike[2], BICs[2] = fit_M3RescorlaWagner_v1(actions, rewards)
    _, loglike[3], BICs[3] = fit_M4CK_v1(actions, rewards)
    _, loglike[4], BICs[4] = fit_M5RWCK_v1(actions, rewards)

    mindex = np.argmin(BICs)
    BEST = np.zeros(5)
    BEST[mindex] = 1

    # not sure yet what the best/sum(best) is about
    # [M, iBEST] = min(BIC);
    # BEST = BIC == M;
    # BEST = BEST / sum(BEST);

    # not sure yet if all these returns will be used
    return BICs, BEST, loglike


exp_trialcount = 1000
bandit = np.array([0.2, 0.8])

# for bla in range(1):
alpha = np.random.uniform(0, 1)
beta = 1 + np.random.exponential(1)
actions, rewards = simulate_M3RescorlaWagner_v1(exp_trialcount, bandit,
                                                alpha, beta)
bics, best, loglikes = fit_all(actions, rewards)
# confusion_matrix[2, :] += best

bics

best
loglikes
alpha
beta

fit_M3RescorlaWagner_v1(actions, rewards)
fit_M5RWCK_v1(actions, rewards)


from LikelihoodFunctions.lik_M5RWCK_v1 import lik_M5RWCK_v1
from LikelihoodFunctions.lik_M3RescorlaWagner_v1 import lik_M3RescorlaWagner_v1
#%%
best_result=[]
best_result_scipy = []
simparams = []

for test in range(100):
    alpha = np.random.uniform(0, 1)
    beta = 1 + np.random.exponential(1)
    # print(f"simulating with alpha: {alpha} and beta: {beta}")
    actions, rewards = simulate_M3RescorlaWagner_v1(exp_trialcount, bandit,
                                                    alpha, beta)
    # likm3 = lik_M3RescorlaWagner_v1([np.float(alpha), np.float(beta)], actions, rewards)
    # alpha
    # beta
    m3_best = np.array([0, 9999, 0, 0])
    for iteration in range(100):
        m3_alpha = np.random.uniform(0, 1)
        m3_beta = np.random.exponential(1)
        likm3 = lik_M3RescorlaWagner_v1([m3_alpha, m3_beta], actions, rewards)
        # print(f"likm3: {likm3} m3a: {m3_alpha} m3b: {m3_beta}")
        if likm3 < m3_best[1]:
            m3_best = np.array([iteration, likm3, m3_alpha, m3_beta])

    paramset, ll, bic = fit_M3RescorlaWagner_v1(actions, rewards)
    best_result_scipy.append([ll, paramset[0], paramset[1]])
    simparams.append([alpha, beta])
    best_result.append(m3_best)

# %%

compare = [(-best_result_scipy[i][0], best_result[i][1]) for i in range(len(best_result))]

scipybest = [sp < brute for sp, brute in compare]

sum(scipybest)

# %%
# from scipy.optimize import minimize
from FittingFunctions.fit_M3RescorlaWagner_v1 import fit_M3RescorlaWagner_v1

best_results_scipy = []
simparams_scipy = []

for test in range(100):
    alpha = np.random.uniform(0, 1)
    beta = 1 + np.random.exponential(1)
    actions, rewards = simulate_M3RescorlaWagner_v1(
        exp_trialcount, bandit, alpha, beta)
    paramset, ll, bic = fit_M3RescorlaWagner_v1(actions, rewards)
    simparams_scipy.append([alpha, beta])
    best_results_scipy.append(paramset)

dist_scipy = [euclidean(simparams_scipy[i], best_results_scipy[i]) for i in range(len(best_results_scipy))]
sns.histplot(dist_scipy)

# %%
SAVE ALL ITERATIONS TO SEE HOW LIKELIHOOD CHANGES WITH ITERATION ETC
meh, do that later, for now lets compare with minimize function

# %%
best_iterations = [row[0] for row in best_result]
import seaborn as sns
sns.histplot(best_iterations)
# %%
from scipy.spatial.distance import euclidean

best_params = [[x[2], x[3]] for x in best_result]
distances = [euclidean(simparams[i], best_params[i]) for i in range(len(best_result))]

sns.histplot(distances)
# %%

alpha_dists = [simparams[i][0] - best_params[i][0] for i in range(len(best_result))]

sns.histplot(alpha_dists)

beta_dists = [simparams[i][1] - best_params[i][1] for i in range(len(best_result))]
sns.histplot(beta_dists)

# %%
    # likm5 = lik_M5RWCK_v1([np.float(alpha), np.float(beta), np.float(0.2), np.float(1.16)], actions, rewards)
    # likm5 = lik_M5RWCK_v1([np.float(alpha), np.float(beta), np.random.uniform(0, 1), np.random.exponential(1)], actions, rewards)
    m5_best = np.array([9999, 0, 0, 0, 0])
    for m5test in range(10):
        likm5 = lik_M5RWCK_v1([np.random.uniform(0, 1), np.random.exponential(1), np.random.uniform(0, 1), np.random.exponential(1)], actions, rewards)

    # print(f"m3: {likm3}")
    # print(f"m5: {likm5}")
    bla.append(likm3 < likm5)

print(f"sum: {sum(bla)}")
