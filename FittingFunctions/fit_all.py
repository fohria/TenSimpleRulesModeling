import numpy as np
from FittingFunctions.fit_M1random import fit_M1random
from FittingFunctions.fit_M2WSLS import fit_M2WSLS
from FittingFunctions.fit_M3RescorlaWagner import fit_M3RescorlaWagner
from FittingFunctions.fit_M4CK import fit_M4CK
from FittingFunctions.fit_M5RWCK import fit_M5RWCK


def fit_all(actions, rewards):

    BICs = np.zeros(5)

    _, _, BICs[0] = fit_M1random(actions, rewards)
    _, _, BICs[1] = fit_M2WSLS(actions, rewards)
    _, _, BICs[2] = fit_M3RescorlaWagner(actions, rewards)
    _, _, BICs[3] = fit_M4CK(actions, rewards)
    _, _, BICs[4] = fit_M5RWCK(actions, rewards)

    mindex = np.argmin(BICs)
    best = np.zeros(5)
    best[mindex] = 1

    # not sure yet if all these returns will be used
    return BICs, mindex, best
