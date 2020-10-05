# so there's a function "simulate_M6RescorlaWagnerBias_v1" but it's never used. at least i can't find that function name anywhere else than in that file, when searching the entire folder structure.

# i'm a bit confused here, the paper makes it sound like we can modify, for example, model3 by adding a bias and thereby capturing some randomness/noise in the simulated data so the two main parameters - alpha and beta - are recovered better.
# but then what they do is simulate a biased model, and show that a model with bias fits better than the model without bias. ... yes? you simulated with bias so how is this interesting? because you're simulating a hypothetical person with such a bias? if so, the bias isn't an "unimportant" parameter, it's actually "there" so the biased model is actually a better model, not "unimportant", no?
# i probably misunderstand again, but wouldn't a more interesting test be to simulate with original M3 and see if the M3+bias helps stabilize recovery of alpha&beta params?

# okay, i'm going to start by doing that because it makes more sense to me and also i'm now curious what will happen..
