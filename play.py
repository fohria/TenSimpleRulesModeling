"""
okay so our plan here will be to start simple and just investigate how the different models work

- [DONE] lets create a conda environment to keep things neat
- overall task: copy matlab code
.
- [DONE] simulate model 1 - random with bias
- [DONE] create analysis_WSLS_v1 function
- [DONE] simulate model2 - noisy wsls
- [DONE] simulate model 3 - RW
- simulate model4
.
.
.
- later task: rewrite everything to be more pythonic and with not stupid variable names. likely also with pandas as some of the matlab for loops would instead become apply functions. and do it literate programming style, so introduce model1 in text with formulas and then the code. and so on. perhaps in notebook style, could test vscode's new notebooks perhaps.
"""

""" MAIN IMPORTS """

%load_ext autoreload
%autoreload 2
#%%
from SimulationFunctions.simulate_M1random_v1 import simulate_M1random_v1
from SimulationFunctions.simulate_M2WSLS_v1 import simulate_M2WSLS_v1
from SimulationFunctions.simulate_M3RescorlaWagner_v1 import simulate_M3RescorlaWagner_v1
from SimulationFunctions.simulate_M4ChoiceKernel_v1 import simulate_M4ChoiceKernel_v1
from AnalysisFunctions.analysis_WSLS_v1 import analysis_WSLS_v1

from numpy import mean
import seaborn as sns

"""
INTRO

so, a participant makes a series of T choices between K slot machines, or ‘one-armed bandits’, to try to maximize their earnings. If played on trial t, each slot machine, k, pays out a reward, rt, which is one with reward probability, μkt, and otherwise 0.

The reward probabilities are different for each slot machine and are initially unknown to the subject. In the simplest version of the task, the reward probabilities are fixed over time.

The three experimental parameters of this task are: the number of trials, T, the number of slot machines, K, and the reward probabilities of the different options, μkt, which may or may not change over time. The settings of these parameters will be important for determining exactly what information we can extract from the experiment. In this example, we will assume that T=1000, K=2, and that the reward probabilities are μ1t=0.2 for slot machine 1 and μ2t=0.8 for slot machine 2.

"""

"""
# BOX1 / Box2/ Figure 2

hmm it's funny they say above K=2 but they just use that implicitly everywhere, i mean i get this is an example but they could've made it a bit more general. but i guess that's something for us to try to do later!



# EXPERIMENT PARAMETERS
"""
# %%

T = 100  # number of trials
mu = [0.2, 0.8]  # mean reward of bandits
"""unfortunate phrasing, as it can be confusing. mu are, according to the main paper text, the reward probabilities not the mean rewards. although over enough trials, since reward is between 0-1 the mean reward will approach the limit of being same as probabilities)"""

# number of repetitions for simulations
Nrep = 110
"""i feel this is also slightly unintuitive. in my mind things get more clear if we call this simulated participants. so instead of seeing this as number of repetitions of simulations, we can see it as 'number of artificial participants' or 'number of fake/simulated participants' or simply 'number of (ro)bots playing'. so i'd probably call this numsims for number of simulations. ideally i'd call it number_of_simulations for full clarity. then you wouldn't need the comment saying it's 'number of repetitions for simulations' ...

another take from here https://stackoverflow.com/questions/3742650/naming-conventions-for-number-of-foos-variables is that we could use sim_count """

# %%
# MODEL1 - RANDOM with bias

sim = [0]  # python needs this to be declared before we can put stuff into it below, the 0 is only a dummy entry in order to get equivalence with matlab numbering so we can use sim[1] for model 1 below insteal of sim[0] for model1
sim.append({"a": [], "r": []})  # this also needs to be setup before adding

for n in range(Nrep):
    b = 0.5
    """ we could put this b variable outside the loop as there's no reason to reassign each time, in python it would not be contained to within the for loop anyway (ie it remains after for loop)

    b for bias is also funny how we can see this is math-first people writing the code. why not just call this variable bias so we can _read_ the code instead of having to _interpret_ it? huh, that's actually a nice catchy line. could become a blog post. 'reading code versus interpreting'"""

    # again here we might as well just call things action and reward to increase readability
    # an additional advantage of using 'action' instead of 'a' is that you can then also use 'actions' to indicate a list/array. below it's unclear at first glance if you get back single actions or multiple from the simulate function.
    a, r = simulate_M1random_v1(T, mu, b)
    sim[1]["a"].append(a)  # python uses 0 indexing wheras matlab start at 1
    sim[1]["r"].append(r)

# %%
# let's test function quickly
# with b=0.5 we should have ~0.5 of each choice
assert b == 0.5 and round(sum([sum(rep) / T for rep in sim[1]['a']]) / Nrep, 1) == 0.5, "something's up with the action selection in random simulation"
# and reward should also be ~0.5
assert b == 0.5 and round(sum([sum(rep) / T for rep in sim[1]['r']]) / Nrep, 1) == 0.5, "something's up with the rewards in random simulation"

# %%

# MODEL 2 : WIN STAY LOSE SHIFT

sim.append({"a": [], "r": []})  # setting up empty lists

for n in range(Nrep):
    epsilon = 0.1
    a, r = simulate_M2WSLS_v1(T, mu, epsilon)
    sim[2]["a"].append(a)
    sim[2]["r"].append(r)

# %%
# MODEL 3 : RESCORLA WAGNER

sim.append({"a": [], "r": []})

for n in range(Nrep):
    alpha = 0.1
    beta = 5
    a, r = simulate_M3RescorlaWagner_v1(T, mu, alpha, beta)
    sim[3]["a"].append(a)
    sim[3]["r"].append(r)

# %%

# MODEL 4 : CHOICE KERNEL

sim.append({"a": [], "r": []})

for n in range(Nrep):
    alpha_c = 0.1
    beta_c = 3
    a, r = simulate_M4ChoiceKernel_v1(T, mu, alpha_c, beta_c)
    sim[4]["a"].append(a)
    sim[4]["r"].append(r)

# %%










# %%
# now we do the WSLS analysis
wsls = []
for i in range(1, len(sim)):
    sim[i]["wsls"] = []  # again, python has to predeclare variables

    for n in range(Nrep):
        sim[i]["wsls"].append(analysis_WSLS_v1(sim[i]["a"][n], sim[i]["r"][n]))

    ls = [sim[i]["wsls"][x][0] for x in range(len(sim[i]["wsls"]))]
    ws = [sim[i]["wsls"][x][1] for x in range(len(sim[i]["wsls"]))]
    wsls.append([mean(ls), mean(ws)])  # again, confusing variable name and positions, i believe it's done for easier plotting 0-1 below but imho that translation should be done when you plot not here. or just name it lsws

#%%

# PLOTTING TIME
sns.set(rc={"figure.figsize": (5, 6), "figure.dpi": 100, "lines.markersize": 15, "lines.linewidth": 3})
sns.set_style("white")
# sns.set_style("ticks")
fig = sns.lineplot(x=[0, 1], y=wsls[0], marker="o", label="M1")
fig = sns.lineplot(x=[0, 1], y=wsls[1], marker="o", label="M2")
fig = sns.lineplot(x=[0, 1], y=wsls[2], marker="o", label="M3")
fig = sns.lineplot(x=[0, 1], y=wsls[3], marker="o", label="M4")
fig.set(ylim=(0, 1), yticks=(0, 0.5, 1), xticks=(0, 1));
fig.set(xlabel="previous reward")
fig.set(ylabel="p(stay)")
fig.set(title="stay behavior")
sns.despine(ax=fig)
