# TenSimpleRulesModeling

This repo contains a Python version of the [original Matlab code](https://github.com/AnneCollins/TenSimpleRulesModeling) for the paper "Ten Simple Rules for Computational Modeling of Psychological Data".

The main files are named according to what box in the paper they describe, i.e. Box1.py, Box2.py etc. These also have corresponding Jupyter notebook files that are generated from the .py files with [Jupytext](https://github.com/mwouts/jupytext):

```bash
jupytext --to notebook Figure2_play.py
```

I made an effort to make the code easy to understand, so in most cases I've changed variable names into more verbose ones. There *should* be comments indicating their equivalent name in the paper. I took this opportunity to practice collating data into the ["tidy" format](http://www.jstatsoft.org/v59/i10/paper), so those parts may not be as neat as they could be.

I've also added comments and clarifications here and there, and some additional analyses where appropriate, as well as confused discussions.

## Installation/Requirements

### simple/browser way

TODO: use notebooks online here (binder?)

### install locally

The main requirements are found in `environment.yml` which can be used to create a new [(ana)conda](https://docs.conda.io/en/latest/) environment like so:

```bash
conda create -f environment.yml
```

### Requirements
The main requirements are:

- numpy
- seaborn
- numba
- pandas

Convenient but not strictly necessary requirements are additionally:

- ipython
- jupytext

## Original Readme
This is the code for the figures in the "Ten Simple Rules for Computational Modeling of Psychological Data" paper.
