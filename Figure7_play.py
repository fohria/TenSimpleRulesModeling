# okay, so here in figure7 we have two models: blind and state-based.
# if i understand the paper text correctly, we simulate the task with both models, and use parameters so their behaviour - in the form of their learning curves - look the same (figA)
# then we fit the state-based (not the blind) to both simulations and see that the likelihood of fit for state-based model ON blind is higher than fit for state-based model ON state-based model (figB)
# but then in figC we see that if we run simulations with the found/fitted parameters for the state-based model and the blind models, we get different behaviours, and the _behaviour_ of the state-based model fits better with figA than the _behaviour_ of the *fitted* blind model to figA.
# easy to get confused here because the colors mean different things in the 3 different plots, but in other words:
# figA: simulations for blind and state-based, lets call them blindSims and stateSims, so this is 'pure' data like plotting behaviour of participants
# figB: likelihood curves for *fitting state-based model* to blindSims and stateSims, so lets call this fitState, i.e. it's one model applied to the two datasets blindSims and stateSims
# figC: learning curves for simulations using fitState and the new fitBlind, so here it's 'pure' data again but now we use the parameter values we found for the two models
