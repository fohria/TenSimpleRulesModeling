# import numpy as np
from numpy.random import random
from .choose import choose

def simulate_M1random_v1(T, mu, b):

    a = []
    r = []

    for t in range(T):

        # compute choice probabilities
        p = [b, 1-b]  # this could be put outside for loop as it doesnt change between trials

        # make choice according to choice probabilities
        # a.append(np.random.choice([0, 1], p=p))
        a.append(choose(p))

        #
        # i never realised i rely on square brackets to signify specific things in python, regular brackets looks like functions and not lists to me!
        r.append(random() < mu[a[t]])

    return a, r
