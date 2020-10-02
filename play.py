from SimulationFunctions.simulate_M3RescorlaWagner_v1 import simulate_M3RescorlaWagner_v1

import numpy as np
from FittingFunctions.fit_M1random_v1 import fit_M1random_v1
from FittingFunctions.fit_M2WSLS_v1 import fit_M2WSLS_v1
from FittingFunctions.fit_M3RescorlaWagner_v1 import fit_M3RescorlaWagner_v1
from FittingFunctions.fit_M4CK_v1 import fit_M4CK_v1
from FittingFunctions.fit_M5RWCK_v1 import fit_M5RWCK_v1


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
