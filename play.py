"""
okay so our plan here will be to start simple and just investigate how the different models work

- [DONE] lets create a conda environment to keep things neat
- overall task: copy matlab code
.
- [DONE] simulate model 1 - random with bias
- create analysis_WSLS_v1 function
.
.
.
- later task: rewrite everything to be more pythonic and with not stupid variable names. likely also with pandas as some of the matlab for loops would instead become apply functions.
"""

""" MAIN IMPORTS """

%load_ext autoreload
%autoreload 2
#%%
from SimulationFunctions.simulate_M1random_v1 import simulate_M1random_v1
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

sim = []  # python needs this to be declared before we can put stuff into it below
sim.append({"a": [], "r": []})  # this also needs to be setup before adding

for n in range(Nrep):
    b = 0.5
    """ we could put this b variable outside the loop as there's no reason to reassign each time, in python it would not be contained to within the for loop anyway (ie it remains after for loop)

    b for bias is also funny how we can see this is math-first people writing the code. why not just call this variable bias so we can _read_ the code instead of having to _interpret_ it? huh, that's actually a nice catchy line. could become a blog post. 'reading code versus interpreting'"""

    # again here we might as well just call things action and reward !!
    a, r = simulate_M1random_v1(T, mu, b)
    sim[0]["a"].append(a)  # python uses 0 indexing wheras matlab start at 1
    sim[0]["r"].append(r)

# %%
# let's test function quickly
# with b=0.5 we should have ~0.5 of each choice
assert b == 0.5 and round(sum([sum(rep) / T for rep in sim[0]['a']]) / Nrep, 1) == 0.5, "something's up with the action selection in random simulation"
# and reward should also be ~0.5
assert b == 0.5 and round(sum([sum(rep) / T for rep in sim[0]['r']]) / Nrep, 1) == 0.5, "something's up with the rewards in random simulation"

# %%

# now we do the WSLS analysis

for i in range(len(sim)):
    sim[i]["wsls"] = []  # again, python has to predeclare variables

    for n in range(Nrep):
        sim[i]["wsls"].append(analysis_WSLS_v1(sim[i]["a"][n], sim[i]["r"][n]))

    ls = [sim[i]["wsls"][i][0] for i in range(len(sim[i]["wsls"]))]
    ws = [sim[i]["wsls"][i][1] for i in range(len(sim[i]["wsls"]))]
    wsls = [mean(ls), mean(ws)]  # again, confusing variable name and positions, i believe it's done for easier plotting 0-1 below but imho that translation should be done when you plot not here. or just name it lsws 

#%%
# PLOTTING TIME
sns.set(rc={"figure.figsize": (4, 6), "figure.dpi": 100, "lines.markersize": 15})
sns.set_style("white")
# sns.set_style("ticks")
fig = sns.lineplot(x=[0, 1], y=wsls, marker="o")
fig.set(ylim=(0, 1), yticks=(0, 0.5, 1), xticks=(0, 1));
fig.set(xlabel="previous reward")
fig.set(ylabel="p(stay)")
fig.set(title="stay behavior")
sns.despine()
