import math, sys
import re
from pathlib import Path
import pandas as pd

import pytest

import numpy as np
np.random.seed(5446)

import warnings

from pygme.factory import model_factory
from pydaisi import daisi_perf, daisi_data

from test_gr2m_update import get_params, NSITES, SITEIDS

def test_get_metricname():

    names = ["ABSFDCFIT100", "ELASTrelRAIN", "ELASTrelEVAP", \
                "NRMSERATIO", "NSELOG", "NSERECIP", "SPLITKGE", \
                "ABSBIAS"]
    for name in names:
        sn = daisi_perf.get_metricname(name, False)
        ln = daisi_perf.get_metricname(name, True)
        assert sn != name



def test_metrics():
    model = model_factory("GR2M")
    defaults = model.params.defaults
    nparams = 3
    params = get_params(nparams, model.params.defaults)

    for isite in range(NSITES):
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(\
                                                        mthly, "per1")
        time = mthly.index[itotal]
        model.allocate(inputs, model.noutputsmax)

        for p in params:
            model.params.values[:model.params.nval] = p
            model.initialise_fromdata()
            model.run()
            sims = model.to_dataframe(include_inputs=True)
            sims.columns = [f"test_{cn}" if not cn in ["Rain", "PET"] else cn\
                                    for cn in sims.columns]
            Qsim = sims.test_Q
            Qobs = 1.1*Qsim
            sims.loc[:, "Qobs"] = Qsim

            perfs = daisi_perf.deterministic_metrics(Qobs, Qsim, \
                        time, ieval, "CAL", "test")

            daisi_perf.elasticity_metrics(model, ieval, "CAL", "test", \
                                perfs)

            assert len(perfs) == 17
            assert all([k.startswith("METRIC") for k in perfs.keys()])


def test_normalised_rmse_ratio(allclose):
    R = 500 # Number of ensembles
    T = 100 # Number of time steps
    # Generate data
    t = np.linspace(0, 1, T)
    y0, y1 = t**3, t**2/5+0.1
    nrepeat = 1000

    # Compute rmse ratios
    nrr = np.zeros((nrepeat, 3))
    for irepeat in range(nrepeat):
        yens = y0[:, None]+y1[:, None]*np.random.normal(size=(T, R))
        yobs = y0+y1*np.random.normal(size=T)
        nrr[irepeat] = daisi_perf.normalised_rmse_ratio(yobs, yens)

    # Check ratio is close to 1
    assert allclose(nrr[:, 0].mean(), 1, atol=1e-2)

