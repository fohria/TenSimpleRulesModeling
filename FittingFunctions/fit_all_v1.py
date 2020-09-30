import numpy as np
from FittingFunctions.fit_M1random_v1 import fit_M1random_v1
from FittingFunctions.fit_M2WSLS_v1 import fit_M2WSLS_v1
from FittingFunctions.fit_M3RescorlaWagner_v1 import fit_M3RescorlaWagner_v1
from FittingFunctions.fit_M4CK_v1 import fit_M4CK_v1
from FittingFunctions.fit_M5RWCK_v1 import fit_M5RWCK_v1


def fit_all_v1(actions, rewards):

    BICs = np.zeros(5)

    _, _, BICs[0] = fit_M1random_v1(actions, rewards)
    _, _, BICs[1] = fit_M2WSLS_v1(actions, rewards)
    _, _, BICs[2] = fit_M3RescorlaWagner_v1(actions, rewards)
    _, _, BICs[3] = fit_M4CK_v1(actions, rewards)
    _, _, BICs[4] = fit_M5RWCK_v1(actions, rewards)

    mindex = np.argmin(BICs)
    BEST = np.zeros(5)
    BEST[mindex] = 1

    # not sure yet what the best/sum(best) is about
    # [M, iBEST] = min(BIC);
    # BEST = BIC == M;
    # BEST = BEST / sum(BEST);

    # not sure yet if all these returns will be used
    return BICs, mindex, BEST
