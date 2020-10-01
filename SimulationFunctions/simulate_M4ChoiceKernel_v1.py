import numpy as np
from numba import njit, int32
from .choose import choose


@njit
def simulate_M4ChoiceKernel_v1(T, mu, alpha_c, beta_c):

    CK = np.array([0.0, 0.0])

    a = np.zeros(T, dtype=int32)
    r = np.zeros(T, dtype=int32)

    for t in range(T):

        # compute choice probabilities
        p = np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK))

        # make choice based on probabilities
        a[t] = choose(np.array([0, 1]), p)

        # generate reward based on choice
        r[t] = np.random.rand() < mu[a[t]]

        # update choice kernel
        # ahh, so the formula is slightly confusing to me: CKkt+1=CKkt+αc(akt−CKkt)
        # but the key is the akt inside parenthesis, it's 0 for every action not chosen, so more simply put than in paper we can say "only the value of the performed action is updated" (which is same as in RW model)
        CK = (1 - alpha_c) * CK
        CK[a[t]] += alpha_c * 1

    return a, r
