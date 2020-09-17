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
    choice = np.random.choice([0, 1, 2])
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

        choice = np.random.choice([0, 1, 2])
        choices.append(choice)

        reward = choice == state % 3
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
df





















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
