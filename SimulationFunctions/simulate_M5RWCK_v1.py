# the standard is to 'import numpy as np' and then use, for example, np.random() but here we want to make the code look kind of similar to matlab
from numpy.random import random
from numpy import exp
from .choose import choose  # matlab adds functions from other files automagically, python needs to import more explicitly

def simulate_M5RWCK_v1(T, mu, alpha, beta, alpha_c, beta_c):

    Q = [0.5, 0.5]
    CK = [0, 0]

    a = []
    r = []

    for t in range(T):

        # compute choice probabilities
        # as our functions become more complex we see how useful it would be to use numpy's linear algebra functionality in order to do V on one line
        # at the same time, perhaps these extra lines make the math more clear to someone more used to python than (linear) algebra
        qs = [beta * q for q in Q]
        cks = [beta_c * ck for ck in CK]
        V = [q + ck for q, ck in zip(qs, cks)]  # V = values? vector? somewhat unclear
        denominator = sum([exp(v) for v in V])
        p = [exp(v) / denominator for v in V]

        # make choice according to choice probabilities
        a.append(choose(p))

        # generate reward based on choice
        r.append(random() < mu[a[t]])

        # update Q values
        delta = r[t] - Q[a[t]]
        Q[a[t]] = Q[a[t]] + alpha * delta

        # update choice kernel
        CK = [(1 - alpha_c) * ck for ck in CK]
        CK[a[t]] = CK[a[t]] + alpha_c * 1

    return a, r
