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
from pydaisi import gr2m_enks, gr2m_update, \
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


def test_compute_sig_and_rho(allclose):

    nval = 1000
    rho = 0.7
    r = np.concatenate([rho**np.arange(nval), \
                            np.zeros(nval-1)])
    M = np.column_stack([np.roll(r, i)[:nval] for i in range(nval)])
    M *= math.sqrt(1-rho**2)

    u = np.random.normal(size=nval)
    se = pd.Series(M.dot(u))

    s0, r0 = gr2m_enks.compute_sig_and_rho(se, 1., 1.)
    assert allclose(s0, 1., atol=5e-2)
    assert allclose(r0, rho, atol=5e-2)

    s, r = gr2m_enks.compute_sig_and_rho(se, 1e-2, 0.)
    assert allclose(s, s0*1e-2)
    assert allclose(r, 0.)

    s, r = gr2m_enks.compute_sig_and_rho(se, 1., 2.)
    assert allclose(r, 1-1e-6)


def test_sigma(allclose):
    nval = 290
    varnames = ["R", "P3", "Q", "P", "S"]
    sigs = {"R": 0.51, "P3": 0.1, "Q": 0.26, "P": 0.13, "S": 11.4}
    rhos = {"R": 0.66, "P3": 0.59, "Q": 0.65, "P": 0.19, "S": 0.71}
    varcorr = np.array([[1, 0.78, 0.79, 0.55, 0.78], \
                        [0.78, 1, 0.78, 0.64, 0.76], \
                        [0.79, 0.78, 1, 0.55, 0.77], \
                        [0.55, 0.64, 0.55, 1, 0.47], \
                        [0.78, 0.76, 0.77, 0.47, 1.]])
    varcorr = pd.DataFrame(varcorr, index=varnames, columns=varnames)
    Sigma, status = gr2m_enks.get_sigma(varnames, sigs, rhos, \
                                varcorr, nval)


def test_sample(allclose):
    nens = 10000
    nval = 500

    sigs = {"A": 1., "B": 3., "C": 0.1}
    rhos = {"A": 0., "B": 0.8, "C": 0.5}
    varnames = ["B", "C", "A"]

    # Check covariance warning is raised
    r1, r2 = 0.9, 0.8
    varcorr = np.array([[1, r1, r2], [r1, 1., r1], [r2, r1, 1]])
    varcorr = pd.DataFrame(varcorr, index=varnames, columns=varnames)
    Sigma, status = gr2m_enks.get_sigma(varnames, sigs, rhos, \
                                varcorr, nval)
    assert status == 1

    # Run sampling with lower covariance that does not trigger warning
    r1, r2 = 0.4, 0.5
    varcorr = np.array([[1, r1, r2], [r1, 1., r1], [r2, r1, 1]])
    varcorr = pd.DataFrame(varcorr, index=varnames, columns=varnames)
    Sigma, status = gr2m_enks.get_sigma(varnames, sigs, rhos, \
                                varcorr, nval)
    assert status == 0

    # Sample
    U = gr2m_enks.sample(Sigma, nens)

    # Check correlation between variables
    nvar = len(sigs)
    V = np.array([U[ivar::nvar, :].ravel() for ivar in range(nvar)])
    co = np.corrcoef(V)
    assert np.allclose(co, varcorr, atol=5e-3, rtol=0.)

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

        # Check auto-regressive component
        R0 = pd.DataFrame(U0).apply(lambda x: x.autocorr())
        R1 = pd.DataFrame(U1).apply(lambda x: x.autocorr())
        assert allclose(R1.mean(), rr, rtol=0., atol=R0.std())
        assert allclose(R0.std(), R1.std(), rtol=5e-2, atol=1e-3)


def test_analysis(allclose):
    # Enks config
    model = gr2m_update.GR2MUPDATE()

    Xr = model.Xr

    sfact = 0.3
    names = ["P", "E", "S", "P3", "R", "Q", "AE", \
                "X1", "X2", "Xr", "Q_obs"]
    stdfacts = {n:sfact for n in names}
    # .. no perturbation of Xr
    stdfacts["Xr"] = 1e-6

    #.. no auto-correlation of perturbations
    rfact = 0.
    rhofacts = {n:rfact for n in names}

    locdur = 5000
    covarfact = 0.1
    clip = 1
    debug = 1
    assim_params = 0

    fimg = FTESTS / "images" / "enks_test"
    fimg.mkdir(exist_ok=True, parents=True)
    # .. clean image folder
    for f in fimg.glob("*.png"):
        f.unlink()

    # parameters
    params = get_params(NSITES, [model.X1, model.X2])

    # Data selection
    nvalmax = 200

    for isite in range(NSITES):
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(mthly, "per1")
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

        # Loop over smoother options
        # .. assim_states = 2 (both states assimilated)
        # .. perturb_states = 3 (both stores)
        # .. perturb_inputs = 3 (both rain and PET)
        # .. assim_params = 0 (no param assim)
        for ensmoother, assim_states, perturb_states, \
                perturb_inputs, assim_params \
                    in prod([1], [2], [3], [3], [0]):

            if perturb_inputs+perturb_states == 0:
                continue
            if assim_params==1 and ensmoother==0:
                continue

            # Initialise enks
            enks = gr2m_enks.EnKS(model, obscal, \
                        stdfacts, rhofacts, covarfact, \
                        locdur, \
                        ensmoother, \
                        perturb_inputs, \
                        perturb_states, \
                        assim_states, \
                        assim_params, \
                        clip, debug)

            enks.initialise()
            enks.plot_dir = fimg
            enks.obs = obs

            # Run smoother
            lab = f"site{isite}_"+\
                        f"ES{ensmoother}_"+\
                        f"PI{perturb_inputs}_PS{perturb_states}_"+\
                        f"AS{assim_states}_AP{assim_params}"
            enks.run(context=lab, message=lab)
            Xa = enks.Xa


