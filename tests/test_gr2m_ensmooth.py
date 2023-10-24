import math, sys, json, os
import re
from itertools import product as prod
from pathlib import Path

import pytest

from timeit import Timer
import time

import numpy as np
np.random.seed(5446)
from scipy.linalg import toeplitz

import pandas as pd

from hydrodiy.stat import sutils, armodels
from hydrodiy.io import csv

import warnings
from pygme.factory import model_factory
from pydaisi import gr2m_ensmooth, gr2m_update, \
                        daisi_utils, \
                        daisi_data

FTESTS = Path(__file__).resolve().parent

from test_gr2m_update import get_params, NSITES, SITEIDS

from tqdm import tqdm

class DUMMY(gr2m_update.GR2MUPDATE):

    def run(self):
        P, E = self.inputs[:, :2].T
        tPerr = self.inputs[:, 2]
        P = np.maximum(0, P+tPerr)

        X1, X2 = self.params.values
        Xr = self.Xr

        # Kalman matrices
        # .. state transition (5 states = P, S, P3, R, Q)
        F = np.zeros((5, 5))
        # .. obs model - 1 obs
        H = np.array([[0.], [0.5]])
        # .. process noise
        #Q = np.array([[
        # .. obs noise

        m = X1/(X1+400)
        P3 = P*m
        S = np.cumsum(P-P3)
        Q = 0.5*P3
        R = np.cumsum(P3-Q)

        onames = self.outputs_names
        iS = onames.index("S")
        self.outputs[:, iS] = S
        iR = onames.index("R")
        self.outputs[:, iR] = R
        iP3 = onames.index("P3")
        self.outputs[:, iP3] = P3
        iQ = onames.index("Q")
        self.outputs[:, iQ] = Q


def test_compute_sig(allclose):
    nval = 1000
    rho = 0.7
    r = np.concatenate([rho**np.arange(nval), \
                            np.zeros(nval-1)])
    M = np.column_stack([np.roll(r, i)[:nval] for i in range(nval)])
    M *= math.sqrt(1-rho**2)

    u = np.random.normal(size=nval)
    se = pd.Series(M.dot(u))

    s0 = gr2m_ensmooth.compute_sig(se, 1.)
    assert allclose(s0, 1., atol=5e-2)

    s = gr2m_ensmooth.compute_sig(se, 1e-2)
    assert allclose(s, s0*1e-2)



def test_sigma(allclose):
    nval = 290
    varnames = ["R", "P3", "Q", "P", "S"]
    sigs = {"R": 0.51, "P3": 0.1, "Q": 0.26, "P": 0.13, "S": 11.4}
    Sigma = gr2m_ensmooth.get_sigma(varnames, sigs, nval)

    nvar = len(varnames)
    n = nvar*nval
    assert Sigma.shape == (n, n)

    d = np.diag(Sigma)
    ii = np.arange(0, n, nvar)
    for i in range(nvar):
        assert allclose(d[ii+i], sigs[varnames[i]]**2)

    assert allclose(Sigma-np.diag(d), 0.)


def test_sample(allclose):
    nens = 10000
    nval = 500

    sigs = {"A": 1., "B": 3., "C": 0.1}
    rhos = {"A": 0., "B": 0.8, "C": 0.5}
    varnames = ["B", "C", "A"]
    nvar = len(varnames)


    # Run sampling with lower covariance that does not trigger warning
    Sigma = gr2m_ensmooth.get_sigma(varnames, sigs, nval)
    U = gr2m_ensmooth.sample(Sigma, nens)

    # Check characteristics for each variable
    for ivar, varname in tqdm(enumerate(varnames), \
                    total=len(varnames), desc="test sampling"):
        # Sample
        ss, rr = sigs[varname], rhos[varname]
        cov = ss**2*toeplitz(rr**np.arange(nval))
        U0 = np.random.multivariate_normal(mean=np.zeros(nval), \
                                    cov=cov, size=nens).T

        # Extract data
        U1 = U[ivar::nvar, :]

        # Check variance
        C0 = np.cov(U0)
        S0 = np.sqrt(np.diag(C0))
        C1 = np.cov(U1)
        S1 = np.sqrt(np.diag(C1))
        assert allclose(S1, S0, rtol=5e-2, atol=0.1)
        assert allclose(S0.std(), S1.std(), rtol=1e-1, atol=5e-2)



def test_analysis(allclose):
    # Enks config
    model = gr2m_update.GR2MUPDATE()
    Xr = model.Xr
    sfact = 0.5
    names = ["P", "E", "S", "P3", "R", "Q", \
                "Q_obs"]
    stdfacts = {n:sfact for n in names}

    # Plotting folder
    fimg = FTESTS / "images" / "ensmooth_test"
    fimg.mkdir(exist_ok=True, parents=True)
    for f in fimg.glob("*.png"):
        f.unlink()

    # parameters
    params = get_params(NSITES, [model.X1, model.X2])

    # Data selection
    nvalmax = 200

    for isite in range(NSITES):
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(\
                                                        mthly, "per1")
        inputs = inputs[-nvalmax:]
        mthly = mthly.iloc[-nvalmax:]

        model.allocate(inputs)
        X1 = params[isite, 0]
        X2 = params[isite, 1]
        model.params.values = [X1, X2, Xr]
        model.initialise_fromdata()
        model.run()

        # Generate Qcal with assimilation one out of 2 obs values
        sims = model.to_dataframe()
        Qobs = sims.Q+np.random.uniform(-1, 1, len(sims))
        Qobscal = np.nan*Qobs.copy()
        ical = np.zeros(len(Qobs))
        ical[::2] = 1
        Qobscal[ical==1] = Qobs.values[ical==1]

        obscal = pd.DataFrame({"Q": Qobscal})
        obs = pd.DataFrame({"Q": Qobs})

        # Corrupt parameters
        X1err = X1*np.random.uniform(-0.1, 0.1)
        X2err = X2*np.random.uniform(-0.1, 0.1)
        model.params.values = [X1+X1err, X2+X2err, Xr]

        # Initialise ensmooth
        ensmooth = gr2m_ensmooth.EnSmooth(model, obscal, \
                    stdfacts, debug=True)

        ensmooth.initialise()
        ensmooth.plot_dir = fimg
        ensmooth.obs = obs

        # Run smoother
        lab = f"site{isite}"
        ensmooth.run(context=lab, message=lab)
        Xa = ensmooth.Xa


