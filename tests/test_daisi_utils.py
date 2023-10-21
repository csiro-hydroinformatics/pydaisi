import math, sys, json
import re
from itertools import product as prod
from pathlib import Path

import pytest

from timeit import Timer
import time

import numpy as np
np.random.seed(5446)

import pandas as pd
from scipy.stats import invgamma

import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import sutils
from pygme.models.gr2m import GR2M

import warnings
from pydaisi import daisi_utils
import c_pydaisi

FTESTS = Path(__file__).resolve().parent

from test_gr2m_modif import get_params, get_data, NSITES

def test_boxcox(allclose):
    # Check continuity of transform
    x, nu = 1, 0.1
    lams = np.linspace(-1, 1, 1000)
    y = np.array([c_pydaisi.boxcox_forward(x, lam, nu) for lam in lams])
    assert (np.abs(np.diff(y))<3e-5).all()

    y = 0.1
    x = np.array([c_pydaisi.boxcox_backward(y, lam, nu) for lam in lams])
    assert (np.abs(np.diff(x))<2e-5).all()

    # Test transforms
    n = 1000
    x = np.linspace(1e-10, 100, n)
    xm = x.reshape((100, 10))
    delta = np.random.uniform(-5, 5, n)

    tx = c_pydaisi.boxcox_forward(1., 2., -2.)
    assert np.isnan(tx)

    for lam in [-1., -0.5, -0.2, 0., 0.2, 0.5, 1.]:
        for nu in [0., 0.1, 10]:
            # Test transform object
            trans = daisi_utils.Transform(lam, nu)
            # .. vectors
            tx = trans.forward(x)
            if abs(lam)>0:
                assert allclose(tx, ((x+nu)**lam-1)/lam)
            else:
                assert allclose(tx, np.log(x+nu))

            x2 = trans.backward(tx)
            assert allclose(x2, x)

            # .. matrices
            txm = trans.forward(xm)
            if abs(lam)>0:
                assert allclose(txm, ((xm+nu)**lam-1)/lam)
            else:
                assert allclose(txm, np.log(xm+nu))

            xm2 = trans.backward(txm)
            assert allclose(xm2, xm)

            # .. perturb
            for xclip in [-nu+1e-4, -nu+1, max(0, -nu+1e-4)]:
                trans.xclip = xclip
                xp = trans.perturb(x, delta)

                txc = trans.forward(trans.xclip)
                if np.isnan(txc):
                    txc = -np.inf

                ymax = np.inf if lam>=0 else -1./lam-1e-2
                y = (trans.forward(x)+delta).clip(txc, ymax)
                xp2 = trans.backward(y)
                assert allclose(xp, xp2, equal_nan=True)

            # Test underlying C functions
            for i in range(n):
                xx = x[i]
                tx = c_pydaisi.boxcox_forward(xx, lam, nu)
                if abs(lam)>0:
                    assert allclose(tx, ((xx+nu)**lam-1)/lam)
                else:
                    assert allclose(tx, math.log(xx+nu))

                xx2 = c_pydaisi.boxcox_backward(tx, lam, nu)
                assert allclose(xx2, xx)

                if lam<1:
                    # Test perturbation - no clipping, i.e. transformed clipped
                    # value = nan
                    dd = delta[i]
                    xclip = -nu
                    assert np.isnan(c_pydaisi.boxcox_forward(xclip, lam, nu))

                    xclip = -nu+1e-4
                    assert not np.isnan(c_pydaisi.boxcox_forward(xclip, lam, nu))

                    # Test pert - clipping
                    xclip = -nu+1e-4
                    y = c_pydaisi.boxcox_forward(xx, lam, nu)+dd
                    yclip = c_pydaisi.boxcox_forward(xclip, lam, nu)
                    xpert = c_pydaisi.boxcox_perturb(xx, dd, lam, nu, xclip)
                    ymax = np.inf if lam>=0 else -1./lam-1e-2
                    expected = c_pydaisi.boxcox_backward(\
                                    min(max(y, yclip), ymax), lam, nu)
                    assert allclose(xpert, expected)


def test_gr2m_fun(allclose):
    """ Compare nonstat with original model when all theta are neutral """
    gr2m = GR2M()
    nparams = 20
    params = get_params(nparams, gr2m.params.defaults)

    minmax = {
        "prod p": [np.inf, 0, 0, -np.inf], \
        "prod e": [np.inf, 0, 0, -np.inf], \
        "prod aen": [np.inf, 0, 0, -np.inf], \
        "prod p3n": [np.inf, 0, 0, -np.inf], \
        "route p3n": [np.inf, 0, 0, -np.inf], \
        "route rstart": [np.inf, 0, 0, -np.inf], \
        "route rend": [np.inf, 0, 0, -np.inf], \
        "route fn": [np.inf, 0, 0, -np.inf], \
        "route qn": [np.inf, 0, 0, -np.inf], \
    }
    def updateminmax(name, x):
        x0, s, n, x1 = minmax[name]
        x0 = min(x0, x.min())
        x1 = max(x1, x.max())
        s += x.sum()
        n += len(x)
        minmax[name] = [x0, s, n, x1]

    for i in range(NSITES):
        _, inputs, _, _, _ = get_data(i)
        gr2m.allocate(inputs, gr2m.noutputsmax)

        for p in params:
            gr2m.params.values[:gr2m.params.nval] = p
            gr2m.initialise_fromdata()
            gr2m.run()
            X1 = gr2m.X1
            X2 = gr2m.X2
            Xr = gr2m.Rcapacity
            df = gr2m.to_dataframe(include_inputs=True)

            S, R = df.S.shift(1), df.R.shift(1)
            Sn, Rn = df.S, df.R
            P, E, P3, Q = df.Rain, df.PET, df.P3, df.Q
            AE, F = df.AE, df.F

            # GR2M forward functions
            Snext = daisi_utils.gr2m_S_fun(X1, S, P, E)
            assert allclose(Snext[1:], df.S.iloc[1:])

            AEhat = daisi_utils.gr2m_AE_fun(X1, S, P, E)
            assert allclose(AEhat[1:], df.AE.iloc[1:])

            P3hat = daisi_utils.gr2m_P3_fun(X1, S, P, E)
            assert allclose(P3hat[1:], df.P3.iloc[1:])

            Rnext = daisi_utils.gr2m_R_fun(X2, Xr, R, P3)
            assert allclose(Rnext[1:], df.R.iloc[1:])

            Qhat = daisi_utils.gr2m_Q_fun(X2, Xr, R, P3)
            assert allclose(Qhat[1:], df.Q.iloc[1:])

            Fhat = daisi_utils.gr2m_F_fun(X2, Xr, R, P3)
            assert allclose(Fhat[1:], df.F.iloc[1:])

            # GR2M state transforms
            u = daisi_utils.gr2m_prod_S_raw2norm(X1, S[1:])
            assert np.all((u>=0) & (u<=1))
            Sb = daisi_utils.gr2m_prod_S_norm2raw(X1, u)
            assert allclose(S[1:], Sb)

            aen = daisi_utils.gr2m_prod_AE_raw2norm(X1, AE)
            updateminmax("prod aen", aen)
            AEb = daisi_utils.gr2m_prod_AE_norm2raw(X1, aen)
            assert allclose(AEb, AE)

            p3n = daisi_utils.gr2m_prod_P3_raw2norm(X1, P3)
            updateminmax("prod p3n", p3n)
            #assert np.all((p3n>=0) & (p3n<=1))
            P3b = daisi_utils.gr2m_prod_P3_norm2raw(X1, p3n)
            assert allclose(P3, P3b)

            pn = daisi_utils.gr2m_prod_P_raw2norm(X1, P)
            updateminmax("prod p", pn)
            #assert np.all((pn>=0) & (pn<=1))
            Pb = daisi_utils.gr2m_prod_P_norm2raw(X1, pn)
            assert allclose(P, Pb)

            en = daisi_utils.gr2m_prod_E_raw2norm(X1, E)
            updateminmax("prod e", en)
            Eb = daisi_utils.gr2m_prod_E_norm2raw(X1, en)
            assert allclose(E, Eb, atol=1e-3, rtol=1e-3)

            p3n = daisi_utils.gr2m_rout_P3_raw2norm(X2, Xr, P3)
            updateminmax("route p3n", p3n)
            P3b = daisi_utils.gr2m_rout_P3_norm2raw(X2, Xr, p3n)
            assert allclose(P3, P3b)

            v = daisi_utils.gr2m_rout_Rstart_raw2norm(X2, Xr, R[1:])
            updateminmax("route rstart", v)
            Rb = daisi_utils.gr2m_rout_Rstart_norm2raw(X2, Xr, v)
            assert allclose(R[1:], Rb)

            v = daisi_utils.gr2m_rout_Rend_raw2norm(X2, Xr, R[1:])
            updateminmax("route rend", v)
            assert np.all((v>=0) & (v<=1))
            Rb = daisi_utils.gr2m_rout_Rend_norm2raw(X2, Xr, v)
            assert allclose(R[1:], Rb)

            fn = daisi_utils.gr2m_rout_F_raw2norm(X2, Xr, F)
            updateminmax("route fn", fn)
            Fb = daisi_utils.gr2m_rout_F_norm2raw(X2, Xr, fn)
            assert allclose(F, Fb)

            qn = daisi_utils.gr2m_rout_Q_raw2norm(X2, Xr, Q)
            updateminmax("route qn", qn)
            Qb = daisi_utils.gr2m_rout_Q_norm2raw(X2, Xr, qn)
            assert allclose(Q, Qb)

            # GR2M forward functions in transform space
            uend = daisi_utils.gr2m_S_fun_normalised(X1, u, pn[1:], en[1:])
            assert allclose(uend[:-1], u[1:], atol=1e-7)

            # .. simple known relationship
            ys, yp, ye = u, pn[1:], en[1:]
            ys1 = (np.tanh(yp)+ys)/(1+np.tanh(yp)*ys)
            ys2 = ys1*(1-np.tanh(ye))/(1+(1-ys1)*np.tanh(ye))
            ysplus = ys2/np.power(1+ys2**3, 1./3)
            assert allclose(uend, ysplus, atol=1e-4)

            p3n_calc = daisi_utils.gr2m_P3_fun_normalised(X1, u, \
                                    pn[1:], en[1:])
            p3n = daisi_utils.gr2m_prod_P3_raw2norm(X1, P3)
            assert allclose(p3n_calc, p3n[1:], atol=5e-4)

            yp3 = yp+ys-ys1+ys2-ysplus
            assert allclose(yp3, p3n_calc, atol=1e-4)

            p3n = daisi_utils.gr2m_rout_P3_raw2norm(X2, Xr, P3)
            v = daisi_utils.gr2m_rout_Rstart_raw2norm(X2, Xr, R[1:])
            vend_calc = daisi_utils.gr2m_R_fun_normalised(X2, Xr, v, p3n[1:])
            vend = daisi_utils.gr2m_rout_Rend_raw2norm(X2, Xr, R[1:])
            assert allclose(vend_calc[:-1], vend[1:])

            sx = v+p3n[1:]
            yrplus = sx/(1+sx)
            assert allclose(yrplus, vend_calc)

            yq_calc = daisi_utils.gr2m_Q_fun_normalised(X2, Xr, v, p3n[1:])
            yq = sx**2/(1+sx)
            assert allclose(yq, yq_calc)

    print("\nMin max norm variables")
    for name, (x0, s, n, x1) in minmax.items():
        xm = s/n
        print(f"    {name:12s} = [{x0:0.2f}, {xm:0.2f}, {x1:0.2f}]")


def test_bayesianreg(allclose):
    # Loop through stan test data
    fstan = FTESTS / "stan_results"

    for i, f in enumerate(fstan.glob("stan_random_*.json")):
        with open(f, "r") as fo:
            jdata = json.load(fo)

        N, K = jdata["N"], jdata["K"]
        Xs = np.array(jdata["Xs"])
        Vtmp = np.array(jdata["V"])
        X = pd.DataFrame(Xs.dot(Vtmp.T))
        X.columns = [f"X{i}" for i in range(1, X.shape[1]+1)]

        # fix V and Xs (minor change in svd due to rounding errors)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt.T
        Xs = U*S[None, :]
        # ....

        y = np.array(jdata["y"])
        beta0 = np.array(jdata["beta0"])
        L0 = np.array(jdata["L0"])
        N0 = jdata["N0"]
        sigsqref = jdata["sigsqref"]
        has_inf_prior = jdata["has_inf_prior"]

        # Fit bayesian reg
        breg = daisi_utils.BayesianRegression(X, y, rcond=0.)
        if has_inf_prior == 1:
            breg.add_prior(beta0, L0, sigsqref, N0)

        breg.solve()

        # Check breg data
        assert allclose(V, breg.V)
        assert allclose(S, breg.S)

        # Check computations
        if has_inf_prior:
            L0m = np.diag(L0)
            tXX = np.diag(S**2)
            tXXi = np.diag(1/S**2)
            etahat = tXXi.dot(Xs.T.dot(y))
            assert allclose(etahat, breg.etahat)

            eta0 = V.T.dot(beta0)
            assert allclose(eta0, breg.eta0)

            Ln = tXX+L0m
            etan = np.linalg.inv(Ln).dot(tXX.dot(etahat)+L0m.dot(eta0))
            assert allclose(etan, breg.etan)

        # Check breg samples
        # .. stan samples
        fp = f.parent / (re.sub("data", "samples", f.stem) + ".csv")
        df, _ = csv.read_csv(fp)
        betas_stan = df.filter(regex="beta", axis=1)
        sigs_stan = np.sqrt(df.sigsq)
        nsamples = len(df)

        # .. breg samples
        betas, sigs = breg.sample(nsamples)
        betas = pd.DataFrame(betas)
        sigs = pd.Series(sigs)

        # .. summary stats
        bm = betas.mean().values
        bs = betas.std().values
        bc = betas.corr().values

        bms = betas_stan.mean().values
        bss = betas_stan.std().values
        bcs = betas_stan.corr().values

        # Tests
        assert allclose(bms, bm, rtol=1e-2, atol=5e-2)
        assert allclose(bss, bs, rtol=1e-2, atol=5e-2)
        assert allclose(bcs, bc, rtol=1e-2, atol=8e-2)


def test_model_elasticity(allclose):
    model = GR2M()
    defaults = model.params.defaults
    nparams = 5
    params = get_params(nparams, model.params.defaults)

    for i in range(20):
        _, inputs, _, _, _ = get_data(i)
        model.allocate(inputs, model.noutputsmax)
        iactive = np.arange(len(inputs))>=24

        for p in params:
            model.params.values[:model.params.nval] = p

            model.initialise_fromdata()
            model.run()

            inputs = model.inputs.copy()
            outputs = model.outputs.copy()

            for relative in [True, False]:
                elast, sims = daisi_utils.model_elasticity(model, iactive, \
                                                            relative)
                assert len(elast) == 2
                assert len(elast["rain"]) == 9
                assert len(elast["evap"]) == 9
                assert (elast["rain"]>=0).all()
                assert (elast["evap"]<=0).all()
                if not relative:
                    assert (elast["rain"]<1.5).all()


                # Check model is not wrecked
                model.initialise_fromdata()
                model.run()
                assert allclose(inputs, model.inputs)
                assert allclose(outputs, model.outputs)


def test_smooth():
    if not daisi_utils.HAS_STATSMODELS:
        pytest.skip("Cannot import statsmodels")

    x = np.linspace(0, 1, 50)
    y = x**2+np.random.uniform(-0.5, 0.5, size=len(x))
    xx, yy = daisi_utils.smooth(x, y)

