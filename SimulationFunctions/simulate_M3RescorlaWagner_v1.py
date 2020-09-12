from numpy.random import random
from numpy import exp
from .choose import choose

def simulate_M3RescorlaWagner_v1(T, mu, alpha, beta):

    Q = [0.5, 0.5]  # init with equal values for each action
    a = []
    r = []

    for t in range(T):

        # compute choice probabilities (softmax)
        denominator = sum([exp(beta * q) for q in Q])
        p = [exp(beta * q) / denominator for q in Q]

        # make choice based on choice probabilities
        a.append(choose(p))

        # generate reward based on choice
        r.append(random() < mu[a[t]])

        # update action values
        delta = r[t] - Q[a[t]]  # in paper this is called prediction error, so it's a bit confusing to call it delta here
        Q[a[t]] = Q[a[t]] + alpha * delta

    return a, r
