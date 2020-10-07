
## TODOs
- can probably integrate figure3 and figure3b into one notebook later

## issues
- OMP Error 15 about llvl and MKL. it may be a macos specific issue: https://stackoverflow.com/a/58869103

## other notes
# using BIC formula from paper (equation 10) for clarity
# i.e. BIC = -2 * log(LL) + km * log(T),
# where km is number of parameters, T is number of trials and LL is loglikelihood
# actually their formula is wrong, the actual formula is:
# BIC = -2 * log(L) + km * log(T)
# in other words, they're using LL both in the formula and the text for the loglikelihood when actually it's log(L), so, their formula SHOULD say:
# BIC = -2 * LL + km * log(T)

### unused files

all M6 files (sim/fit/lik) are never used
SimulationFunctions/simulate_validationModel_v1.m

figure6 and figure7 files are self contained and dont use the sim/fit/lik folders at all. especially fig7 has very confusing one letter variable names that don't always correlate with names in the paper.

i suspect someone started making everything neat and then got stressed and/or the two authors divided code for figures and each did it their way.

## describe the problem/bandit setup
More specifically, we consider the case in which a participant makes a series of T choices between K slot machines, or ‘one-armed bandits’, to try to maximize their earnings. If played on trial t, each slot machine, k, pays out a reward, rt, which is one with reward probability, μkt, and otherwise 0. The reward probabilities are different for each slot machine and are initially unknown to the subject. In the simplest version of the task, the reward probabilities are fixed over time.

The three experimental parameters of this task are: the number of trials, T, the number of slot machines, K, and the reward probabilities of the different options, μkt, which may or may not change over time. The settings of these parameters will be important for determining exactly what information we can extract from the experiment. In this example, we will assume that T=1000, K=2, and that the reward probabilities are μ1t=0.2 for slot machine 1 and μ2t=0.8 for slot machine 2.
