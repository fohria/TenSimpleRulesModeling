"""

TODO:

- [DONE] read up on experimental task in paper
- go through matlab code .. in matlab :)


- recreate fig3/box3; gonna be tedious as authors don't actually define the model in appendix4 they mainly refer to another paper. which is not open access.
.
.

EXPERIMENT TASK:

okay so task is basically; you see a picture and have three buttons to press: find the correct button by trial and error. ns = setsize/number of stimuli, i.e. how many pictures were there to learn.

"six blocks in which nS = 2, four blocks in which nS = 3, and three blocks each of nS=4,5,or6 for a total of 19 blocks, and a maximum of 50 min."

"""

""" BOX3/FIG3 """

%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import pandas as pd
# %%
"""
why is this model not included in the "SimulationFunctions"? it's a bit weird they've picked a completely different experimental task and model for this figure compared to the previous figure.
"""

# lets simulate above described task
realalpha = 0.1
realbeta = 10
realrho = 0.9
realK = 4

# %%
"""
    now code starts
"""

# lets start with set size ns=2 and just one loop through the 15x2 trials
ns = 2

# initialize WM (working memory) mixture weight
w = realrho * np.min([1, realK / ns])

# initialize RL and WM "agents" (i'd personally say "action values" or something along those lines)
Q = 0.5 + np.zeros([ns, 3])
WM = 0.5 + np.zeros([ns, 3])

# we use tile instead of repeat so we get [0,1,0,1] instead of [0,0,1,1], in order to have the order similar to matlab code
trials = np.tile(np.arange(ns), 15)
# i would perhaps also shuffle the array but maybe that doesn't matter for the algorithm. in the c&f 2012 paper they do have "pseudorandom" presentations within a block but unclear what exactly that means

choices = []
rewards = []

# enumerate will automatically create an index for each stimulus in the trials array; (0,0), (1,1), (2,0), (3,1)
for trial, state in enumerate(trials):

    # RL policy
    # no need to calculate the same thing twice like in the matlab code. not that it's likely to cause much of a difference in speed for this small problem, but it lowers the risk of typing errors (aka brainfarts where you write beta on one line and realbeta on the other)
    softmax1_value = np.exp(realbeta * Q[state])  # we could also do Q[state, :] like in matlab. matlab however would with Q(3) for example give the single value on first row, second column
    softmax1 = softmax1_value / np.sum(softmax1_value)

    # WM policy
    # i assume this is the simplification for this paper compared to c&f2012; instead of a beta value for WM policy we just use a high value here to make it greedy, i.e. basically pick the highest value all the time
    softmax2_value = np.exp(50 * WM[state])
    softmax2 = softmax2_value / np.sum(softmax2_value)

    # mixture policy
    probabilities = (1-w) * softmax1 + w * softmax2

    # action choice
    # numpy has a really nice function to make a random choice between X choices
    choice = np.random.choice([0, 1, 2], p=probabilities)
    choices.append(choice)

    # reward correct action (arbitrarily defined)
    # rem is matlabs mod, and we adapt to 0-indexing
    reward = choice == state % 3
    rewards.append(reward)

    # update Q and WM values
    Q[state, choice] += realalpha * (reward - Q[state, choice])
    WM[state, choice] = reward

# %%

# okay! now do the loops for set sizes and repetitions of each set size. save as pandas table, that'll make it easy to save and later access by label instead of keeping track of what's what

# lets create a function for the above model
def RLWM(rho, setsize, K, beta, alpha):

    w = rho * np.min([1, K / setsize])
    Q = 0.5 + np.zeros([setsize, 3])
    WM = 0.5 + np.zeros([setsize, 3])

    trials = np.tile(np.arange(setsize), 15)

    choices = []
    rewards = []

    for state in trials:

        # RL policy
        softmax1_value = np.exp(beta * Q[state])
        softmax1 = softmax1_value / np.sum(softmax1_value)

        # WM policy
        softmax2_value = np.exp(50 * WM[state])
        softmax2 = softmax2_value / np.sum(softmax2_value)

        # mixture policy
        probabilities = (1-w) * softmax1 + w * softmax2

        choice = np.random.choice([0, 1, 2], p=probabilities)
        choices.append(choice)

        reward = choice == (state % 3)
        rewards.append(reward)

        Q[state, choice] += alpha * (reward - Q[state, choice])
        WM[state, choice] = reward

    return trials, choices, rewards

# %%
# create a dataframe to hold our data
column_names = ["reward", "choice", "stimulus", "setsize", "block", "trial"]
df = pd.DataFrame(columns=column_names)

block = -1  # to match 0-indexing and keep track of block

for rep in range(3):
    for ns in range(2,7):  # up to but not including 7
        block += 1
        stimuli, choices, rewards = RLWM(realrho, ns, realK, realbeta, realalpha)
        assert len(stimuli) == len(choices) and len(choices) == len(rewards)
        data = np.array([
            rewards, choices, stimuli,
            np.repeat(ns, len(stimuli)),
            np.repeat(block, len(stimuli)),
            np.arange(len(stimuli))
        ])
        # such horrible naming here haha
        df2 = pd.DataFrame(columns=column_names, data=data.transpose())
        df = df.append(df2, ignore_index=True)


# %%
for ns in range(2, 7):
    print(df[df["setsize"] == ns]["reward"].mean())
# %%
df[df["setsize"] == ns]["reward"]

# %%

# okay, so to calculate likelihood it's basically the same function as before we just return likelihood instead of choices and rewards, and input the relevant columns from existing dataframe.

# column_names = ["reward", "choice", "stimulus", "setsize", "block", "trial"]

# so to just briefly explain what happens, is that we go through all the trials and the choices at each point. we get what our model says the probabilities are that each action is correct, say [0.1 0.7 0.2]. if we pick the middle option, 0.7, then that's:
np.log(0.7)
# and if we pick the left option then that's
np.log(0.1)
# so say the recorded choice on this trial was the middle one. the probability for that choice is , according to our model, high. and you see there's almost a difference of 2 between the log numbers of these two choices, so over many trials we would get a large difference between a series of choices that are likely according to our model, compared to a model where the discrepancies keep being wrong. (okay this can be explained more clearly, but we can rewrite it later)


def likelihood_RLWM(rho, K, beta, alpha, blockdata):

    # to access based on trial index starting at 0 we create arrays from df
    trials  = np.array(blockdata["stimulus"])
    rewards = np.array(blockdata["reward"])
    choices = np.array(blockdata["choice"])
    setsize = np.array(blockdata["setsize"])[0]  # only need 1

    w = rho * np.min([1, K / setsize])
    Q = 0.5 + np.zeros([setsize, 3])
    WM = 0.5 + np.zeros([setsize, 3])

    loglikelihood = 0

    for trial, state in enumerate(trials):

        # RL policy
        softmax1_value = np.exp(beta * Q[state])
        softmax1 = softmax1_value / np.sum(softmax1_value)

        # WM policy
        softmax2_value = np.exp(50 * WM[state])
        softmax2 = softmax2_value / np.sum(softmax2_value)

        # mixture policy
        probabilities = (1-w) * softmax1 + w * softmax2

        # so now, based on probabilities calculated what's the probability of the choice that was made?
        prob_choice = probabilities[choices[trial]]
        loglikelihood += np.log(prob_choice)

        choice = choices[trial]
        reward = rewards[trial]
        Q[state, choice] += alpha * (reward - Q[state, choice])
        WM[state, choice] = reward

    return loglikelihood

# %%

alphas = np.arange(0.05, 1.05, 0.05)  # i have to say matlab syntax is nicer here as it includes the upper number so you don't have to make it look this weird
betas = np.append([1], np.arange(5, 55, 5))
rhos = np.arange(0, 1.05, 0.05)
Ks = np.arange(2,7)

loglikes = []

from itertools import product

combos = product(alphas, betas, rhos, Ks)

for alpha, beta, rho, K in product(alphas, betas, rhos, Ks):
    loglike_allblocks = 0
    for block in np.unique(df['block']):
        print(block)
        loglike_allblocks += likelihood_RLWM(rho, K, beta, alpha,
                                             df[df['block'] == block])
    loglikes.append([alpha, beta, rho, K, loglike_allblocks])

fit_results = pd.DataFrame(columns=["alpha", "beta", "rho", "K", "loglike"], data=loglikes)

# %%
# code above took 15 mins which is way more than matlab. hmm actually, cant we do an apply function across instead? oh but we've to do it per block ..hmm..
# anyway save data so we can use it to plot without having to rerun in case kernel has to be restarted.
fit_results.to_csv('fitresults.csv')

fit_results.head()
# %%

# fig A : heatmap of alpha and rho as per figure in paper

fit_results.head(10)[['alpha', 'rho', 'loglike']]
fit_results.head(25)[['alpha', 'rho', 'loglike']]pivot_table(
    values='loglike', index='alpha', columns='rho',
    fill_value=0, aggfunc='mean')
fit_results.head(25)[['alpha', 'rho', 'loglike']].pivot_table(
    values='loglike', index='alpha', columns='rho', aggfunc='mean')

hotdata = fit_results[['alpha', 'rho', 'loglike']].pivot_table(values='loglike', index='alpha', columns='rho', aggfunc='mean')

sns.heatmap(hotdata)

# okay from the looks of that we should just remove rho=1 to get a more nuanced view since we are looking for the "light" values anyway

fits_no_1rho = fit_results[['alpha', 'rho', 'loglike']][fit_results[['alpha', 'rho', 'loglike']]['rho'] != 1]

sns.heatmap(fits_no_1rho.pivot_table(values='loglike', index='alpha', columns='rho', aggfunc='mean'))

# now we can more clearly see that it indeed looks like the summed log likelihoods indicate that alpha=0.05 and rho is somewhere between 0.7 and 0.95, i.e. the top right of the heatmap. so let's zoom in even more on the data shall we

zoomed_in = fit_results[['alpha', 'rho', 'loglike']].query('rho > 0.5 and rho < 1')
zoomed_in = zoomed_in.query('alpha < 0.15')

sns.heatmap(zoomed_in.pivot_table(values='loglike', index='alpha', columns='rho', aggfunc='mean'))

# well, it looks like our likelihood function really nailed it! that's actually not very common. and weirdly the matlab code isn't great at finding it, it is sometimes quite far away.
# above was written when i thought alpha was 0.05 but realalpha is actually 0.1 :P


# fig B : matlab code also has 1d plots for each variable with likelihood on y axis, we could try a seaborn facetgrid for that

fit_results.head()

sns.set(rc={"figure.figsize": (5, 6), "figure.dpi": 100, "lines.markersize": 5, "lines.linewidth": 3})
sns.set_style("white")

sns.pairplot(fit_results, x_vars=['alpha', 'beta', 'rho', 'K'], y_vars=['loglike'])


test = fit_results[['alpha', 'beta', 'loglike', 'K','rho']]
sns.heatmap(test.pivot_table(values='loglike', index='alpha', columns='beta', aggfunc='mean'))
test.pivot_table(values='loglike', index=['alpha','K'], columns=['beta', 'rho'], aggfunc='mean')

sns.pairplot(test.pivot_table(values='loglike', index=['alpha','K'], columns=['beta', 'rho'], aggfunc='mean'))

import pandas as pd
import seaborn as sns
fit_results = pd.read_csv('fitresults2.csv')
# okay so we have wide data format
testdata = fit_results.head(35)
pivtable = testdata.pivot_table(values='loglike', index=['alpha', 'K'], columns=['beta', 'rho'], aggfunc='mean')

# there's no reason really for having alpha and K as index , it just seemed neat to have two variables as index and two as columns
fit_pivot = fit_results.pivot_table(values='loglike', index=['alpha', 'K'], columns=['beta', 'rho'], aggfunc='mean')

fit_pivot.head()

# with this we can construct a huge heatmap/clustermap combining all the parameters
sns.clustermap(fit_pivot)

# it's a bit difficult to see patterns here, as it automagically creates cluster linkages, i.e. the dendritic trees, so we can do it without the clustering:
sns.clustermap(fit_pivot, row_cluster=False, col_cluster=False)
import seaborn as sns
sns.clustermap(fit_pivot, row_cluster=False, col_cluster=False)

# it's still maybe not the easiest to read the values here, but if you check the values on the top-right  for example you see it says 0.05-2 meaning we have alpha=0.05 and K=2. we then have 5 little squares below that where K increases to its max of 6 (it says 0.05-5 but that's just the axis label, there's a square under that for K=6). then alpha increases to 0.1 and we have K from 2 to 6 again. and so on.

# well then, now we can clearly see the the bottom right values are pretty crap overall. we can also see that we have a pattern of "boxes" , the repeats of K and rho, and in those boxes we see that they also have a pattern of lower right values being kinda crap. so in general, high values of K are unlikely.

fit_results['loglike'].std()
fit_results['loglike'].median()
fit_results['loglike'].max()
fit_results['loglike'].min()

#sim2
fit_results['loglike'].std()
fit_results['loglike'].median()
fit_results['loglike'].max()
fit_results['loglike'].min()

# let's remove all results where loglike > mean (well, lower than mean actually)
fit_mean = fit_results['loglike'].mean()
fit_zoomed = fit_results.query('loglike > @fit_mean')  # query is convenient but slightly confusing as you need to use @ to indicate variables in the query
fit_zoom_pivot = fit_zoomed.pivot_table(values='loglike', index=['alpha', 'K'], columns=['beta', 'rho'], aggfunc='mean')

sns.clustermap(fit_zoom_pivot, row_cluster=False, col_cluster=False)

# haha that didn't really help
fit_results[['alpha', 'loglike']].groupby('alpha').aggregate(np.mean)
sns.set()
sns.lineplot(data=fit_results[['alpha', 'loglike']].groupby('alpha').aggregate(np.mean))
sns.lineplot(data=fit_results[['beta', 'loglike']].groupby('beta').aggregate(np.mean))
sns.lineplot(data=fit_results[['rho', 'loglike']].groupby('rho').aggregate(np.mean))
sns.lineplot(data=fit_results[['K', 'loglike']].groupby('K').aggregate(np.mean))

import numpy as np
import pandas as pd
import seaborn as sns
fit_results = pd.read_csv('fitresults2.csv')
fit_results['loglike'].max()
fit_results.query('loglike > -262')
# we can use fitresults3 and ... 1? 2? to compare, in one case we have exact correct parameters and in the 3 case we have a bunch of them really close in loglike but quite varied ranges on alpha and beta , rho and K are pretty close though. but how would you know if you didnt have the real values? you dont.
# this is in my opinion quite unofrtunate that is rarely mentioned in papers except as sidenotes or short comments, like the daw tutorial paper "can be quite fiddly" or whatever it says. lets see what they say in this paper.
fit_results.to_csv('fitresults3.csv', index=False)
fit_results[['alpha', 'loglike']].query('loglike > -262')

# huh, well, the sucker actually got it exactly right ..

"""
    ah! i wonder if it got it exactly right because we did that check above about having decreasingly bad performance with increasing set size. if i remember correctly, our data had 'perfect' decrease, so potentially the likelihood calculations will be more off if we have a not so perfect decrease correlation with setsize. so we are now running again to check.
"""

# so, a funny thing is that it now warns the matrix is huge and recommends installing a package called fastcluster:
# /Users/foh/miniconda3/envs/tensimplerules/lib/python3.8/site-packages/seaborn/matrix.py:649: UserWarning: Clustering large matrix with scipy. Installing `fastcluster` may give better performance.
# let's install that and see what happens!
# first just check that it takes around 2 secs to create the above clustermap
# after installation and restart of kernel we have no warning, but it's barely faster. maybe we would see bigger difference for much larger datasets.





test.groupby(['alpha', 'beta', 'rho', 'K']).mean()

sns.pairplot(test.groupby('loglike'))





















# %%
# SCRATCHPAD

rows = []
rows.append([1,2,3,4,5])
rows.append([6,7,8,9,10])

loglikes.append(rows)

np.transpose([4,5,6,7])
loglikes.append(pd.Series([1, 2, 3, 4, 55]), ignore_index=True)


from itertools import product
[x for x in product([0,1],[3,4],[6,7],[8,9])]
df









#df[df['block'] == 1]['setsize'].iloc[0]

#likelihood_RLWM(realrho, realK, realbeta, realalpha, df[df['block'] == 1])

df[df['block'] == 1]['setsize'].iloc[44]
blockdata["setsize"]

df[df['block'] == 1]



np.repeat(4, 5)
df['reward'] = [1,2,3,5]

df

bla = pd.DataFrame(columns=column_names, data=[[1,2,3,4,5,6]])

bla

reward = [1,1,1,1]
choice = [2,2,2,2]
stimulus = [3,3,3,3]
setsize=[4,4,4,4]
block=[5,5,5,5]
trial=[6,6,6,6]

hey = np.array([reward, choice, stimulus, setsize, block, trial])

bla2 = pd.DataFrame(columns=column_names, data=hey.transpose())

df.append(bla2)
df
# then we can experiment with multiindex pandas to keep track of all likelihoods perhaps? good practice
rew = choice == (state % 3)
1 % 3

np.sum(np.exp(10 * np.array([0.5, 0.5, 0.55])))

np.exp(10 * np.array([0.5, 0.5, 0.55])) / sum(np.exp(10 * np.array([0.5, 0.5, 0.55])))

bla = np.exp(10 * np.array([0.5, 0.5, 0.55]))
bla

bla / np.sum(bla)

np.exp(beta * Q[state])
softmax1 = softmax1_value / np.sum(softmax1_value)
