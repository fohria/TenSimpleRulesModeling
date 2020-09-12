from numpy.random import random
from numpy import exp
from .choose import choose

def simulate_M4ChoiceKernel_v1(T, mu, alpha_c, beta_c):

    CK = [0, 0]

    a = []
    r = []

    for t in range(T):

        # compute choice probabilities
        denominator = sum([exp(beta_c * ck) for ck in CK])
        p = [exp(beta_c * ck) / denominator for ck in CK]

        # make choice based on probabilities
        a.append(choose(p))

        # generate reward based on choice
        r.append(random() < mu[a[t]])

        # update choice kernel
        # ahh, so the formula is slightly confusing to me: CKkt+1=CKkt+αc(akt−CKkt)
        # but the key is the akt inside parenthesis, it's 0 for every action not chosen, so more simply put than in paper we can say "only the value of the performed action is updated" (which is same as in RW model)
        CK = [(1 - alpha_c) * ck for ck in CK]
        CK[a[t]] = CK[a[t]] + alpha_c * 1

    return a, r
