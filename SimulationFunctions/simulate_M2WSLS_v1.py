from numpy.random import random
from .choose import choose

def simulate_M2WSLS_v1(T, mu, epsilon):

    # last reward/action initialized to None
    rLast = None
    aLast = None

    # initialize the action and reward lists
    a = []
    r = []

    for t in range(T):

        # compute choice probabilities
        if rLast is None:

            # choose randomly on first trial
            p = [0.5, 0.5]

        else:

            # choice depends on last reward
            if rLast == 1:

                # winstay (with probability 1-epsilon)
                # again we've to use list comprehensions when we don't use numpy or pandas https://stackoverflow.com/questions/35166633/how-do-i-multiply-each-element-in-a-list-by-a-number
                p = [epsilon / 2 * x for x in [1, 1]]
                p[aLast] = 1 - epsilon / 2

            else:

                # lose shift (probability 1-epsilon)
                p = [(1 - epsilon / 2) * x for x in [1, 1]]
                p[aLast] = epsilon / 2

        # make choice according to choice probabilities
        a.append(choose(p))

        # generate reward based on choice
        r.append(random() < mu[a[t]])

        # set last action and reward to current before next loop iteration
        aLast = a[t]
        rLast = r[t]

    return a, r
