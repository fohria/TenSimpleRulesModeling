import numpy as np
from scipy.optimize import minimize
from LikelihoodFunctions.lik_M5RWCK_v1 import lik_M5RWCK_v1


def fit_M5RWCK_v1(actions, rewards):

    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
    bounds = [(1e-10, 1 - 1e-10), (1e-10, 20),
              (1e-10, 1 - 1e-10), (1e-10, 20)]

    fit_success = False
    max_iterations = 10
    # alpha, alphac ~ uniform(0,1)
    # beta, betac ~ exponential(1)
    # TODO: i guess ideally these guesses should be declared as optional parameters to this function to allow more flexibility
    # parameter order: alpha, beta, alphac, betac
    while fit_success is False and max_iterations != 0:
        start_guess = [np.random.uniform(0, 1), np.random.exponential(1),
                       np.random.uniform(0, 1), np.random.exponential(1)]
        fitresult = minimize(
            lik_M5RWCK_v1, start_guess, args=(actions, rewards), bounds=bounds)
        fit_success = fitresult.success
        max_iterations -= 1

    parameter_estimations = fitresult.x
    loglikelihood = -fitresult.fun

    trial_count = len(actions)
    BIC = -2 * loglikelihood + 1 * np.log(trial_count)

    return parameter_estimations, loglikelihood, BIC


def fit_M5RWCK_parallel(actions, rewards):

    from optimparallel import minimize_parallel  # putting this here until we use it

    # alpha, alphac ~ uniform(0,1)
    # beta, betac ~ exponential(1)
    # TODO: i guess ideally these guesses should be declared as optional parameters to this function to allow more flexibility
    # parameter order: alpha, beta, alphac, betac
    start_guess = [np.random.uniform(0, 1), np.random.exponential(1),
                   np.random.uniform(0, 1), np.random.exponential(1)]
    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
    bounds = [(1e-10, 1 - 1e-10), (1e-10, 20),
              (1e-10, 1 - 1e-10), (1e-10, 20)]

    fitresult = minimize_parallel(lik_M5RWCK_v1,
                                  start_guess,
                                  args=(actions, rewards),
                                  bounds=bounds)

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
