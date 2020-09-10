from numpy import mean

def analysis_WSLS_v1(a, r):

    aLast = [None] + a[:-1]
    # aLast == a  # without pandas library we cant do this
    stay = [aLast[i] == a[i] for i in range(len(a))]  # comprehensions to the rescue
    rLast = [None] + r[:-1]

    # again we are not using pandas so rely on comprehensions
    winStay = mean([stay[i] for i in range(len(rLast)) if rLast[i] == 1])
    loseStay = mean([stay[i] for i in range(len(rLast)) if rLast[i] == 0])

    return loseStay, winStay  # confusing to call function wsls and return lsws


"""
function out = analysis_WSLS_v1(a, r)

aLast = [nan a(1:end-1)];  # "shift" array
stay = aLast == a;  # compare where last action == current action
rLast = [nan r(1:end-1)];  # shift reward array

winStay = nanmean(stay(rLast == 1));  # get all items from stay with indices of rlast == 1, i.e. when we have action stay and last reward 1 we won and will be staying
loseStay = nanmean(stay(rLast == 0));
out = [loseStay winStay];
"""
