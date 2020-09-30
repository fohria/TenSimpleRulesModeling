import numpy as np
from scipy.optimize import minimize
from LikelihoodFunctions.lik_M2WSLS_v1 import lik_M2WSLS_v1

def fit_M2WSLS_v1(actions, rewards):

    start_guess = [np.random.uniform(0, 1)]
    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    bounds = [(1e-10, 1 - 1e-10)]

    fitresult = minimize(lik_M2WSLS_v1, start_guess, args=(actions, rewards), bounds=bounds)

    parameter_estimations = fitresult.x
    loglikelihood = -fitresult.fun
    # using BIC formula from paper (equation 10) for clarity
    # i.e. BIC = -2 * log(LL) + km * log(T),
    # where km is number of parameters, T is number of trials and LL is loglikelihood
    # actually their formula is wrong, the actual formula is:
    # BIC = -2 * log(L) + km * log(T)
    # in other words, they're using LL both in the formula and the text for the loglikelihood when actually it's log(L), so, their formula SHOULD say:
    # BIC = -2 * LL + km * log(T)
    trial_count = len(actions)
    BIC = -2 * loglikelihood + 1 * np.log(trial_count)

    return parameter_estimations, loglikelihood, BIC
