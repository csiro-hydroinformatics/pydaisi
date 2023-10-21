import math, sys, json
import re, os
from itertools import product as prod
from pathlib import Path
from inspect import signature

import pytest

from timeit import Timer
import time

import numpy as np
np.random.seed(5446)
import pandas as pd

import matplotlib.pyplot as plt

from pygme.factory import model_factory
from hydrodiy.io import csv
from hydrodiy.stat import metrics
from hydrodiy.plot import putils

import c_pydaisi
from pydaisi import gr2m_update, daisi_utils, daisi_data

import warnings

from tqdm import tqdm

FTESTS = Path(__file__).resolve().parent

SITEIDS = [405218]
NSITES = len(SITEIDS)

def get_params(nparams, defaults):
    X1 = np.clip(defaults[0]*np.random.uniform(0.2, 5, nparams), 1, 10000)
    X2 = np.clip(defaults[1]+np.random.uniform(-0.5, 0.5, nparams), 0.1, 10)
    return np.column_stack([X1, X2])


def test_get_interp_params_ini(allclose):
    # params names
    names = gr2m_update.get_interp_params_names()
    assert len(names["S"]) == 10
    assert len(names["P3"]) == 10
    assert len(names["R"]) == 6
    assert len(names["Q"]) == 6

    # params values
    X1 = 400.
    X2 = 0.8
    Xr = 100.
    lamP, lamE, lamQ, nu = 0., 1., 0.2, 1.
    pp = gr2m_update.get_interp_params_ini(X1, X2, Xr, \
                                lamP=lamP, lamE=lamE, lamQ=lamQ, \
                                nu=nu)
    ini = pp["params"]
    for k, v in ini.items():
        if re.search("GR2M", k):
            continue
        assert (v==0).all()

    assert len(ini["S"]) == 10
    assert len(ini["P3"]) == 10
    assert len(ini["R"]) == 6
    assert len(ini["Q"]) == 6
    assert len(pp["GR2M"]) == 3


def test_fit(allclose):
    model = model_factory("GR2M")
    nmodel = gr2m_update.GR2MUPDATE()
    nparams = 3
    params = get_params(nparams, model.params.defaults)
    lamP, lamE, lamQ, nu = 0., 1., 0.2, 0.2

    for isite in range(NSITES):
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(mthly, "per1")
        nval = len(inputs)
        model.allocate(inputs, model.noutputsmax)
        nmodel.allocate(inputs, nmodel.noutputsmax)
        ical = np.arange(nval)>24

        for iparam, p in enumerate(params):
            X1, X2 = p
            model.X1 = X1
            model.X2 = X2
            model.initialise_fromdata()
            sini = model.states.values.copy()

            nmodel.X1 = X1
            nmodel.X2 = X2
            nmodel.Xr = 60.
            nmodel.lamP = lamP
            nmodel.lamE = lamE
            nmodel.lamQ = lamQ
            nmodel.initialise_fromdata()

            # Get original simulation (independently from modif)
            model.run()
            sims = model.to_dataframe(include_inputs=True)

            # Get original simulation
            simsb = nmodel.get_parent_simulation(sini)
            assert allclose(sims, simsb)

            interp_data = gr2m_update.get_interpolation_variables(simsb, \
                                    nmodel.X1, nmodel.X2, nmodel.Xr, \
                                    nmodel.alphaP, nmodel.alphaE, \
                                    useradial=0, uselinpred=1, \
                                    useconstraint=0)
            dd = interp_data["data"]
            assert dd["XS"].shape == (nval, 3)
            assert dd["WS"].shape == (nval, 27)
            assert dd["XR"].shape == (nval, 2)
            assert dd["WR"].shape == (nval, 9)

            for vname in ["Yuend", "Yp3n", "Yvend", "Yqn"]:
                assert dd[vname].shape[0] == nval

            # Noisy data
            simsb_noisy = simsb.copy()*np.random.normal(loc=1,scale=0.3, \
                                                    size=simsb.shape)
            interp_data_noisy = gr2m_update.get_interpolation_variables(simsb_noisy, \
                                    nmodel.X1, nmodel.X2, nmodel.Xr, \
                                    nmodel.alphaP, nmodel.alphaE, \
                                    useradial=0, uselinpred=1, \
                                    useconstraint=0)

            # Run fitting
            betans = {}
            betahats = {}
            for rcond, usemean in prod(rconds, [0, 1]):
                # we must get all parameters
                # close to 0 if rcond is reasonable.
                if rcond>=1e-3:
                    fit = gr2m_update.fit_interpolation(\
                                        interp_data, \
                                        lamP=lamP, lamE=lamE, lamQ=lamQ,\
                                        nu=nu, \
                                        ical=ical, \
                                        rcondS=rcond, \
                                        rcondR=rcond, \
                                        usebaseline=1, \
                                        usemean=usemean)
                    for k, v in fit["params"].items():
                        if k == "GR2M":
                            continue
                        assert allclose(0, v, rtol=0, atol=5e-2)

                # Adding perturbations to fitting data
                # to get params != 0
                fit = gr2m_update.fit_interpolation(\
                                    interp_data_noisy, \
                                    lamP=lamP, lamE=lamE, lamQ=lamQ,\
                                    nu=nu, \
                                    ical=ical, \
                                    rcondS=rcond, \
                                    rcondR=rcond, \
                                    usebaseline=1, \
                                    usemean=usemean)

                fitparams = fit["params"]
                for state in fitparams.keys():
                    if state == "GR2M":
                        continue

                    pp = fitparams[state]

                    nparams = 27 if state in ["S", "P3"] else 9
                    assert pp.shape[0] == nparams

                    # Check prior covariance
                    breg = fit["info"][f"{state}_diag"]["breg"]
                    assert breg.has_inf_prior

                    St = breg.S.max()*rcond
                    assert allclose(breg.L0, St**2)

                    # Check computation
                    L0, tXX = breg.L0, breg.S**2
                    eta0, etahat = breg.eta0, breg.etahat
                    etan = (tXX*etahat+L0*eta0)/(tXX+L0)
                    assert allclose(etan, breg.etan)

                    key = (state, rcond, usemean)
                    betans[key] = breg.betan
                    betahats[key] = breg.V.dot(breg.etahat)

            # Check differences in beta
            for state in fitparams.keys():
                bn = np.array([betans[(state, rc, 1)] for rc in rconds])
                bh = np.array([betahats[(state, rc, 1)] for rc in rconds])

                # Check all beta hat are the same
                # (independant from rcond)
                assert allclose(np.diff(bh, axis=0), 0.)

                # Check decreasing difference (to a certain approx level)
                # associated with decreasing rcond
                # (i.e. lower and lower importance of prior leading
                #  to betan being closer and closer to betahat
                #  which is the uninformative prior esimate)
                d = np.diff(np.abs(bn-bh).mean(axis=1))
                assert np.all(d<0)



def test_modif_compare_with_original(allclose):
    gr2m = model_factory("GR2M")

    # First approx model not using baseline
    nmodel = gr2m_update.GR2MUPDATE()
    nmodelprod = gr2m_update.GR2MUPDATE()

    # Site selection
    isites = np.arange(NSITES).tolist()
    nparams = 20
    for isite in isites:
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(mthly, "per1")
        params = get_params(nparams, gr2m.params.defaults)

        for iparam, (X1, X2) in enumerate(params):
            nmodel.X1 = X1
            nmodel.X2 = X2
            nmodel.set_interp_params()
            nmodel.allocate(inputs, nmodel.noutputsmax)

            nmodelprod.X1 = X1
            nmodelprod.X2 = X2
            nmodelprod.set_interp_params()
            pp = nmodelprod.get_interp_params()
            # .. modifiy routing functions but NOT production
            pp["params"]["R"] = np.random.uniform(-1, 1, 9)
            pp["params"]["Q"] = np.random.uniform(-1, 1, 9)
            nmodelprod.set_interp_params(pp)
            nmodelprod.allocate(inputs, nmodel.noutputsmax)

            gr2m.X1 = X1
            gr2m.X2 = X2
            gr2m.allocate(inputs, gr2m.noutputsmax)
            gr2m.initialise_fromdata()
            nmodel.states.values = gr2m.states.values.copy()
            nmodelprod.states.values = gr2m.states.values.copy()

            nmodel.run()
            nmodelprod.run()
            sims = nmodel.to_dataframe()
            simsp = nmodelprod.to_dataframe()

            gr2m.run()
            gsims = gr2m.to_dataframe()
            cc = list(set(sims.columns) & set(gsims.columns))
            sims, simsp, gsims = sims.loc[:, cc], simsp.loc[:, cc], gsims.loc[:, cc]
            diff = np.abs(sims-gsims)
            maxval = diff.max()
            imax = diff.loc[:, diff.columns[maxval>1e-4]].idxmax()
            assert allclose(sims, gsims, atol=1e-3, rtol=1e-3)

            # Same production simulations
            cprod = ["S", "AE", "P3"]
            assert allclose(simsp.loc[:, cprod], gsims.loc[:, cprod], \
                                    atol=1e-3, rtol=1e-3)
            # .. but not routing
            crout = ["Q", "F", "R"]
            assert not np.allclose(simsp.loc[:, crout], gsims.loc[:, crout], \
                                    atol=1e-3, rtol=1e-3)


def test_modif(allclose):
    gr2m = model_factory("GR2M")

    # First approx model not using baseline
    nmodel = gr2m_update.GR2MUPDATE()

    lamP, lamE, lamQ, nu = 0., 1., 0.2, 0.2
    nmodel.lamP = lamP
    nmodel.lamE = lamE
    nmodel.lamQ = lamQ
    nmodel.nu = nu

    nparams = 10
    Xr = gr2m.Rcapacity

    # interpolation parameters
    rcondS = 0.01
    rcondR = 0.01

    # Site selection
    isites = np.arange(NSITES).tolist()
    tbar = tqdm(total=len(isites)*nparams, \
                    desc="comparing modif")
    res = []
    for isite in isites:
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(mthly, "per1")
        params = get_params(nparams, nmodel.params.defaults)

        gr2m.allocate(inputs, gr2m.noutputsmax)
        nmodel.allocate(inputs, nmodel.noutputsmax)

        ical = np.arange(len(inputs))>=12*3

        for iparam, p in enumerate(params):
            tbar.update()
            X1, X2 = p
            X1 = min(10000, X1)

            alphaP, alphaE = 1., 1.

            # GR2M simul
            gr2m.X1 = X1
            gr2m.X2 = X2
            gr2m.initialise_fromdata()
            sini = gr2m.states.values.copy()
            gr2m.run()
            sims = gr2m.to_dataframe(include_inputs=True)

            # Set config data and initialise
            nmodel.X1 = X1
            nmodel.X2 = X2
            nmodel.Xr = Xr
            nmodel.initialise(sini)

            # Use noisy simulations to get lin approx params != 0
            # (otherwise we get GR2M model structure)
            psims = nmodel.get_parent_simulation(sini)
            cc = [cn for cn in psims.columns if re.search("S|P3|R|Q", cn)]
            noise = np.random.normal(loc=1, scale=0.1, size=(psims.shape[0], len(cc)))
            psims.loc[:, cc] = psims.loc[:, cc]*noise

            # Set interp param for model without baseline
            interp_data = gr2m_update.get_interpolation_variables(psims, \
                                    X1, X2, Xr, \
                                    alphaP, alphaE, \
                                    useradial=0, \
                                    uselinpred=1, \
                                    useconstraint=0, \
                                    radius=0.)

            thetas_ref = gr2m_update.fit_interpolation(\
                                            interp_data, \
                                            lamP=lamP, lamE=lamE, lamQ=lamQ,\
                                            nu=nu, \
                                            ical=ical, \
                                            usebaseline=1, \
                                            rcondS=rcondS, \
                                            rcondR=rcondR, \
                                            usemean=1)
            nmodel.set_interp_params(thetas_ref)

            # Check interp param setup
            thetas = nmodel.get_interp_params()["params"]
            for state in ["S", "P3", "R", "Q"]:
                assert allclose(thetas[state], thetas_ref["params"][state])
                th = nmodel.config.to_series()\
                            .filter(regex=f"^{state}").values
                assert allclose(thetas[state], th)

            # Run new model
            nmodel.run()
            simsm = nmodel.to_dataframe()

            # Reproduce modif computations
            # when not using baseline
            nval = len(sims)
            cols = ["S", "P3", "AE", "F", "R", "Q", \
                        "yS_baseline", "yP3_baseline", "yR_baseline", "yQ_baseline", \
                        "yS_update", "yP3_update", "yR_update", "yQ_update"]
            S, R = sini
            simsmb = pd.DataFrame(np.nan, index=np.arange(nval), \
                                            columns=cols)
            w2 = np.zeros(9)
            w3 = np.zeros(27)

            for i in range(nval):
                # Production
                Sstart = S
                P = inputs[i, 0]
                E = inputs[i, 1]

                # .. baseline variables
                S_baseline = daisi_utils.gr2m_S_fun(X1, [S], [P], [E])
                u_baseline = daisi_utils.gr2m_prod_S_raw2norm(X1, S_baseline)[0]

                P3_baseline = daisi_utils.gr2m_P3_fun(X1, [S], [P], [E])
                p3n_baseline = daisi_utils.gr2m_prod_P3_raw2norm(X1, P3_baseline)[0]

                # .. modified variables
                z = np.array([\
                        daisi_utils.gr2m_prod_S_raw2norm(X1, [S])[0], \
                        daisi_utils.gr2m_prod_P_raw2norm(X1, [P])[0], \
                        daisi_utils.gr2m_prod_E_raw2norm(X1, [E])[0] \
                ])
                w3[:10] = [1., z[0], z[1], z[2], \
                                z[0]**2, z[1]**2, z[2]**2,
                                z[0]*z[1], z[0]*z[2], \
                                z[1]*z[2]]

                ths = thetas["S"]
                u_modif = w3.dot(ths)
                S = daisi_utils.gr2m_prod_S_norm2raw(X1, [u_baseline+u_modif])[0]

                thp3 = thetas["P3"]
                p3n_modif = w3.dot(thp3)
                P3 = daisi_utils.gr2m_prod_P3_norm2raw(X1, [p3n_baseline+p3n_modif])[0]

                # .. bounds and mass balance
                S = max(0., min(min(Sstart+P, X1), S))
                simsmb.loc[i, "S"] = S

                P3 = max(0., min(P+Sstart-S, P3))
                simsmb.loc[i, "P3"] = P3

                AE = P-S+Sstart-P3
                simsmb.loc[i, "AE"] = AE

                # Routing
                # .. baseline variables
                Rstart = R
                R_baseline = daisi_utils.gr2m_R_fun(X2, Xr, [Rstart], [P3])
                v_baseline = daisi_utils.gr2m_rout_Rend_raw2norm(X2, Xr, R_baseline)[0]

                Q_baseline = daisi_utils.gr2m_Q_fun(X2, Xr, [Rstart], [P3])
                qn_baseline = daisi_utils.gr2m_rout_Q_raw2norm(X2, Xr, Q_baseline)[0]

                # .. modified variables
                z = np.array([\
                        daisi_utils.gr2m_rout_Rstart_raw2norm(X2, Xr, [Rstart])[0], \
                        daisi_utils.gr2m_rout_P3_raw2norm(X2, Xr, [P3])[0] \
                ])
                thr = thetas["R"]
                w2[:6] = [1, z[0], z[1], \
                                z[0]**2, z[1]**2, \
                                z[0]*z[1]]
                v_modif = w2.dot(thr)
                R = daisi_utils.gr2m_rout_Rend_norm2raw(X2, Xr, \
                                            [v_baseline+v_modif])[0]

                thq = thetas["Q"]
                qn_modif = w2.dot(thq)
                Q = daisi_utils.gr2m_rout_Q_norm2raw(X2, Xr, [qn_baseline+qn_modif])[0]

                # .. bounds and mass balance
                R = max(0., min(Xr, R))
                Q = max(0., Q)
                simsmb.loc[i, "R"] = R
                simsmb.loc[i, "Q"] = Q
                simsmb.loc[i, "F"] = R-Rstart-P3+Q

                simsmb.loc[i, "yS_baseline"] = u_baseline
                simsmb.loc[i, "yP3_baseline"] = p3n_baseline
                simsmb.loc[i, "yR_baseline"] = v_baseline
                simsmb.loc[i, "yQ_baseline"] = qn_baseline


                simsmb.loc[i, "yS_update"] = u_modif
                simsmb.loc[i, "yP3_update"] = p3n_modif
                simsmb.loc[i, "yR_update"] = v_modif
                simsmb.loc[i, "yQ_update"] = qn_modif

            # Check sim
            assert (simsm.S>=-1e-8).all()
            assert (simsm.S<=X1+-1e-8).all()
            assert (simsm.R>=-1e-8).all()
            assert (simsm.R<=Xr+-1e-8).all()
            assert (simsm.AE>=-1e-8).all()
            assert (simsm.P3>=-1e-8).all()

            # Check modif simulation against python loop
            cc = simsmb.columns
            simsmc, simsmbc = simsm.loc[:, cc], simsmb.loc[:, cc]
            diff = np.abs(simsmc-simsmbc)
            imax = diff.idxmax()
            assert allclose(simsmc, simsmbc, atol=1e-3, rtol=1e-2)



def test_perturbation(allclose):
    nmodel = gr2m_update.GR2MUPDATE()
    nparams = 10
    Xr = nmodel.Xr

    # interpolation parameters
    lamP, lamE, lamQ, nu = 0., 0.5, 0.2, 0.7
    nmodel.lamP = lamP
    nmodel.lamE = lamE
    nmodel.lamQ = lamQ
    nmodel.nu = nu

    # transforms
    transP = nmodel.transP
    transP.xclip = 0.
    transE = nmodel.transE
    transE.xclip = 0.
    transQ = nmodel.transQ
    transQ.xclip = 0.

    # Site lists
    isites = np.arange(NSITES).tolist()
    tbar = tqdm(total=len(isites)*nparams, desc="comparing random")
    res = []
    for isite in isites:
        mthly = daisi_data.get_data(SITEIDS[isite])
        inputs, obs, itotal, iactive, ieval = daisi_data.get_inputs_and_obs(mthly, "per1")
        params = get_params(nparams, nmodel.params.defaults)
        nval = len(inputs)
        pert = np.random.normal(size=(nval, 6))*1.
        pert[:, 2] = 0.
        inputs = np.column_stack([inputs, pert])
        nmodel.allocate(inputs, nmodel.noutputsmax)

        ical = np.arange(len(inputs))>=12*3

        for iparam, p in enumerate(params):
            tbar.update()
            X1, X2 = p
            X1 = min(10000, X1)
            Xr = np.random.uniform(50, 70)

            # Set config data and initialise
            nmodel.params.values = [X1, X2, Xr]
            nmodel.initialise_fromdata()
            sini = nmodel.states.values.copy()

            psims = nmodel.get_parent_simulation(sini)
            interp_data = gr2m_update.get_interpolation_variables(psims, \
                                    X1, X2, Xr)

            interp_params = gr2m_update.fit_interpolation(\
                                            interp_data, \
                                            lamP=lamP, lamE=lamE, lamQ=lamQ,\
                                            nu=nu, \
                                            ical=ical)

            nmodel.set_interp_params(interp_params)
            thetas = interp_params["params"]

            # Run modif model with perturb
            nmodel.run()
            msims = nmodel.to_dataframe().copy()

            # Reproduce modif model computations
            nval = len(msims)
            S, R = sini
            cols = ["S", "E", "P3", "R", "F", "Q", "P"]
            msimsb = pd.DataFrame(np.nan, index=np.arange(nval), \
                                            columns=cols)
            w2 = np.zeros(9)
            w3 = np.zeros(27)

            for i in range(nval):
                # Inputs
                P = transP.perturb(alphaP*inputs[i, 0], pert[i, 0])
                E = transE.perturb(alphaE*inputs[i, 1], pert[i, 1])

                msimsb.loc[i, "P"] = P
                msimsb.loc[i, "E"] = E

                # States
                Sini = np.clip(S+pert[i, 2], 0, X1)
                Rini = np.clip(R+pert[i, 4], 0, Xr)

                ### Production ###
                # .. baseline variables
                S_baseline = daisi_utils.gr2m_S_fun(X1, [Sini], [P], [E])
                u_baseline = daisi_utils.gr2m_prod_S_raw2norm(X1, S_baseline)[0]

                P3_baseline = daisi_utils.gr2m_P3_fun(X1, [Sini], [P], [E])
                p3n_baseline = daisi_utils.gr2m_prod_P3_raw2norm(X1, P3_baseline)[0]

                # .. modif
                x = np.array([\
                        daisi_utils.gr2m_prod_S_raw2norm(X1, [Sini])[0], \
                        daisi_utils.gr2m_prod_P_raw2norm(X1, [P])[0], \
                        daisi_utils.gr2m_prod_E_raw2norm(X1, [E])[0] \
                ])
                w3[0] = 1.
                w3[1] = x[0]
                w3[2] = x[1]
                w3[3] = x[2]
                w3[4] = x[0]**2
                w3[5] = x[1]**2
                w3[6] = x[2]**2
                w3[7] = x[0]*x[1]
                w3[8] = x[0]*x[2]
                w3[9] = x[1]*x[2]
                u_modif = w3.dot(thetas["S"])
                S = daisi_utils.gr2m_prod_S_norm2raw(X1, [u_baseline+u_modif])[0]

                thp3 = thetas["P3"]
                p3n_modif = w3.dot(thp3)
                P3 = daisi_utils.gr2m_prod_P3_norm2raw(X1, [p3n_baseline+p3n_modif])[0]

                # Check and mass balance
                S = max(0., min(min(Sini+P, X1), S))
                msimsb.loc[i, "S"] = S

                P3 = max(0., min(P+Sini-S, P3))

                # .. perturb
                P3 = transP.perturb(P3, pert[i, 3])
                msimsb.loc[i, "P3"] = P3

                AE = P+Sini-S-P3
                msimsb.loc[i, "AE"] = AE

                ### Routing ###

                # .. baseline variables
                R_baseline = daisi_utils.gr2m_R_fun(X2, Xr, [Rini], [P3])
                v_baseline = daisi_utils.gr2m_rout_Rend_raw2norm(X2, Xr, \
                                                                R_baseline)[0]

                Q_baseline = daisi_utils.gr2m_Q_fun(X2, Xr, [Rini], [P3])
                qn_baseline = daisi_utils.gr2m_rout_Q_raw2norm(X2, Xr, \
                                                                Q_baseline)[0]

                x = np.array([\
                        daisi_utils.gr2m_rout_Rstart_raw2norm(X2, Xr, [Rini])[0], \
                        daisi_utils.gr2m_rout_P3_raw2norm(X2, Xr, [P3])[0] \
                ])
                w2[0] = 1.
                w2[1] = x[0]
                w2[2] = x[1]
                w2[3] = x[0]**2
                w2[4] = x[1]**2
                w2[5] = x[0]*x[1]
                v_modif = w2.dot(thetas["R"])
                R = daisi_utils.gr2m_rout_Rend_norm2raw(X2, Xr, \
                                                    [v_baseline+v_modif])[0]
                qn_modif = w2.dot(thetas["Q"])
                Q = daisi_utils.gr2m_rout_Q_norm2raw(X2, Xr,\
                                                    [qn_baseline+qn_modif])[0]
                # .. bounds and mass balance
                R = max(0., min(Xr, R))
                msimsb.loc[i, "R"] = R

                Q = max(0., Q)

                # perturb
                Q = transQ.perturb(Q, pert[i, 5])
                msimsb.loc[i, "Q"] = Q

                msimsb.loc[i, "F"] = R-Rini-P3+Q

            cc = list(set(msimsb.columns) & set(msims.columns))
            assert allclose(msims.loc[:, cc], \
                            msimsb.loc[:, cc], rtol=5e-2)

