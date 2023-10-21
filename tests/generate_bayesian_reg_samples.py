#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2023-02-14 Tue 11:58 AM
## Comment : pystan example
##           see https://pystan.readthedocs.io/en/latest/getting_started.html
##
## ------------------------------
from pathlib import Path
import json, sys

import numpy as np
np.random.seed(5446)

import pandas as pd
from scipy.interpolate import splev, splrep
from cmdstanpy import CmdStanModel

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
N, K = 300, 20

nchains = 5
nwarm = 10000
nsamples = 50000
nrepeat = 10


##----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
FTESTS = source_file.parent
stan_file = FTESTS / "stan_regression.stan"

stan_folder = FTESTS / "stan_outputs"
stan_res = FTESTS / "stan_results"

#
for d in [stan_folder, stan_res]:
    d.mkdir(exist_ok=True)
    for f in d.glob("*.*"):
        f.unlink()

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
x = np.logspace(-1, 0, K)   # Log spacing to reduce points towars 1 and
                            # increasing ill-conditionning of matrix X
x = np.concatenate([x, [2]])
u = np.linspace(x[0], x[-1], N)

X = np.zeros((N, K))
for i in range(K):
    y = np.zeros_like(x)
    y[i] = 1.
    spl = splrep(x, y)
    X[:, i] = splev(u, spl)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
# Transform data
U, S, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T
Xs = U.dot(np.diag(S))
assert np.allclose(Xs.dot(V.T), X)

# build stan model
model =  CmdStanModel(stan_file=stan_file)

for irepeat in range(nrepeat):
    # Random data
    beta = np.random.uniform(-1, 1, K)
    y0 = X.dot(beta)
    sig = np.std(y0)*0.5
    y = y0+np.random.normal(size=N, scale=sig)

    # Inference config
    sigsqref = np.random.uniform(0, y.std()/2)**2
    beta0 = np.random.uniform(-0.5, 0.5, K)
    N0 = 2.
    St = np.random.choice(S[3:K-2])
    L0 = St**2*np.ones(K)
    has_inf_prior = irepeat > 1 # no prior for first two sims only

    # Stan Data
    stan_data = {"N": N, "K": K, \
                "Xs": Xs, \
                "V": V, \
                "S": S, \
                "y": y, \
                "beta0":beta0, \
                "L0": L0, \
                "N0": N0, \
                "sigsqref": sigsqref, \
                "has_inf_prior": has_inf_prior}

    # Sample from model
    smp = model.sample(data=stan_data, \
                        chains=nchains, \
                        iter_warmup=nwarm, \
                        iter_sampling=nsamples//nchains, \
                        output_dir=stan_folder, \
                        show_progress=True)
    df = smp.draws_pd()
    p = df.filter(regex="beta|sigsq", axis=1)

    # Store
    fd = stan_res / f"stan_random_data_{irepeat}.json"
    for n in ["Xs", "S", "V", "y", "beta0", "L0"]:
        stan_data[n] = stan_data[n].tolist()

    with open(fd, "w") as fo:
        json.dump(stan_data, fo, indent=4)

    fp = stan_res / f"stan_random_samples_{irepeat}.csv"
    csv.write_csv(p, fp, "Stan samples", source_file, float_format="%0.4f")


LOGGER.info("Process completed")
