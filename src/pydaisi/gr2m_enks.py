import re, math
from pathlib import Path
from itertools import combinations_with_replacement as combwr
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, solve

import matplotlib.pyplot as plt
from matplotlib.dates import date2num


from pydaisi.daisi_utils import Transform
import c_pydaisi

from tqdm import tqdm

FHERE = Path(__file__).resolve().parent
FROOT = FHERE.parent


def ar1chol(nval, rho):
    z = rho**np.arange(nval)
    M = np.concatenate([z, z[::-1][:-1]])[None, :]
    M = np.row_stack([np.roll(M, i, axis=1) \
                        for i in range(nval)])[:, :nval].T
    C = np.linalg.cholesky(M)
    return C


def get_sigma(varnames, sigs, rhos, variabcorr, nval):
    nvar = len(varnames)
    assert all([(n in sigs) for n in varnames]), \
                        "Expected all varnames in sigs"
    assert all([(n in rhos) for n in varnames]), \
                        "Expected all varnames in sigs"

    variabcorr = variabcorr.loc[varnames, varnames].values
    assert variabcorr.shape == (nvar, nvar), \
                        "Expected rhovar.shape=(nvar, nvar)."
    np.linalg.cholesky(variabcorr)
    assert np.allclose(variabcorr, variabcorr.T)
    assert np.allclose(np.diag(variabcorr), 1.)

    # Compute cholesky decomposition of covariance matrix
    # for each variable (i.e. time covariance)
    # .. original naive form
    #C = [sigs[vn]*np.linalg.cholesky(\
    #            toeplitz(rhos[vn]**np.arange(nval))) \
    #                for vn in varnames]
    # .. more stable form
    C = [sigs[vn]*ar1chol(nval, rhos[vn]) \
                        for vn in varnames]

    # Build covariance matrix by brute force,
    # there may be a much smarter way to do this...
    Sigma = np.zeros((nval*nvar, nval*nvar))
    status = 0
    indices = np.arange(nval*nvar)
    for i, j in combwr(range(nvar), 2):
        # Reorganise matrix to have each columns of ens containing:
        # [ v0[t0], v1[t0], ..., vn[t0], v0[t1], v1[t1], ...
        #                                   vn[t1], v1[t2],...]
        ii = indices[i::nvar][:, None]
        vi = varnames[i]
        vj = varnames[j]

        if i==j:
            S = C[i].dot(C[i].T)
            Sigma[ii, ii.T] = S
        else:
            B0 = C[i].dot(C[j].T)

            # Asymptotic value of diagonal term of B0
            ri, rj = rhos[vi], rhos[vj]
            si, sj = sigs[vi], sigs[vj]
            vref = si*sj*math.sqrt(1-ri**2)*math.sqrt(1-rj**2)/(1-ri*rj)

            # Target covariance vi/vj
            v = variabcorr[i, j]*si*sj

            # Set threshold to avoid degenerescence
            vthresh = vref*0.999
            if v>vthresh:
                mess = f"Reducing covariance {vi}/{vj} to keep matrix "+\
                        f"semi-definite positive (v={v:0.4f}/"+\
                        f"vthresh={vthresh:0.4f})"
                warnings.warn(mess)
                status = 1

            v = min(v, vthresh)
            B = v/vref*B0

            jj = np.arange(nval*nvar)[j::nvar][None, :]
            Sigma[ii, jj] = B
            Sigma[jj.T, ii.T] = B.T

    eig, Q = np.linalg.eig(Sigma)

    if np.any(np.iscomplex(eig)):
        r, im = eig.real, eig.imag
        assert np.all(np.abs(im)<1e-10)
        eig = r

    if np.any(eig<-1e-10):
        nc = np.sum(eig<-1e-10)
        mess = f"Correcting {nc}/{len(eig)} eigen values "+\
             f"to keep Sigma def pos. min(eig)/max(eig) = "+\
             f"{eig.min()/eig.max():2.2e}."
        warnings.warn(mess)
        status = 1
        eig = np.maximum(eig, 0)

    Sigma = Q.dot(np.diag(eig)).dot(Q.T)

    return Sigma, status


def sample(Sigma, nens):
    eig, Q = np.linalg.eig(Sigma)
    U = np.random.normal(size=(Sigma.shape[0], nens))
    return Q.dot(np.diag(np.sqrt(eig))).dot(U)


def compute_sig_and_rho(x, stdfact, rhofact):
    assert stdfact>=1e-6 and stdfact<=1.
    assert rhofact>=0. and rhofact<=2.

    sig = np.nanstd(x)*stdfact # ensures sig>1e-6
    sig = 1e-3 if np.isnan(sig) or sig<1e-3 else sig

    rho = pd.Series(x).autocorr()
    if rhofact <= 1.0:
        rho = rho*rhofact
    else:
        f = rhofact-1
        rho = max(rho, 1-1e-6)*f+rho*(1-f)

    rho = 0 if np.isnan(rho) or rho<0 else rho

    return sig, rho


def compute_corr(x, variabcorrfact):
    xnames = x.columns.tolist()
    x = np.array(x)
    assert x.ndim == 2
    assert x.shape[0]//2>x.shape[1]
    assert variabcorrfact>=0 and variabcorrfact<=2

    iok = np.all(~np.isnan(x), axis=1)
    x = x[iok]
    nvar = x.shape[1]

    co = np.corrcoef(x.T)
    d = np.diag(co)
    d = np.where((d>1e-6)&~np.isnan(d), d, 1e-6)
    co[np.isnan(co)] = 0.
    k = np.arange(nvar)
    co[k, k] = d
    # Check covariance is ok
    np.linalg.cholesky(co)

    if variabcorrfact<=1:
        e = np.eye(nvar)
        corr = e+variabcorrfact*(co-e)
    else:
        e = np.eye(nvar)
        o = e+(np.ones((nvar, nvar))-e)*0.99
        a = variabcorrfact-1
        corr = a*o+(1-a)*co

    # Check corr matrix is ok
    np.linalg.cholesky(corr)

    corr = pd.DataFrame(corr, \
                    index=xnames, \
                    columns=xnames)
    return corr



class EnKS():
    def __init__(self, model, \
                    obscal, \
                    stdfacts, rhofacts, variabcorrfact, \
                    locdur, \
                    ensmoother, \
                    perturb_inputs_code, \
                    perturb_variables_code, \
                    assim_states_code, \
                    assim_params_code, \
                    clip, debug, nens=200):

        # Analysed variables clipping
        assert clip in [0, 1]

        # Run EnS and not EnKS
        assert ensmoother in [0, 1]
        self.ensmoother = ensmoother
        # Dims
        self.nens = nens

        # Perturb inputs:
        # 0 No input perturbations
        # 1 Rainfall only
        # 2 PET only
        # 3 Rainfall and PET
        assert perturb_inputs_code in [0, 1, 2, 3]

        # Perturb states:
        # 0 No states perturbations
        # 1 Production states only
        # 2 Routing states only
        # 3 Both production and routing
        assert perturb_variables_code in [0, 1, 2, 3]

        # We need to perturb something within the model!
        assert perturb_inputs_code+perturb_variables_code>0

        # List of perturbed states
        perturb_variables = []
        if perturb_inputs_code in [1, 3]:
            perturb_variables.append("P")

        if perturb_inputs_code in [2, 3]:
            perturb_variables.append("E")

        if perturb_variables_code in [1, 3]:
            perturb_variables.extend(["S", "P3"])

        if perturb_variables_code in [2, 3]:
            perturb_variables.append("R")

        # Always perturb outputs otherwise filter becomes unstable
        perturb_variables.append("Q")

        self.perturb_variables = perturb_variables

        # Assimilated states
        # X0  assimilate production states
        # X1  assimilate routing states
        # X2  assimilate both
        assert assim_states_code in [0, 1, 2, 10, 11, 12]
        assim_states = []
        if assim_states_code%10 in [0, 2]:
            assim_states.extend(["S", "P3"])

        if assim_states_code%10 in [1, 2]:
            assim_states.append("R")

        # <10 -> do not assimilate inputs
        # >=10 -> assimilate inputs
        if assim_states_code//10 == 1:
            assim_states.extend(["P", "E"])

        # Assimilate parameters
        assert assim_params_code in [0, 1]
        # .. cannot assimilate parameters when using EnKS
        # .. (need time varying parameters)
        assert not ((assim_params_code==1) and (ensmoother==0))

        self.assim_params = assim_params_code
        if assim_params_code == 1:
            self.nparams = model.params.values.shape[0]
        else:
            self.nparams = 0

        # GR2M params
        self.X1 = model.X1
        self.Xr = model.Xr
        self.X2 = model.X2

        # Transforms
        xclip = 1e-3 if clip==1 else -model.nu+2e-4
        self.transP = Transform(model.lamP, model.nu, xclip)
        self.transQ = Transform(model.lamQ, model.nu, xclip)
        self.transE = Transform(model.lamE, model.nu, xclip)

        # Obs data
        assert isinstance(obscal, pd.DataFrame), "expected obscal to be pd.DataFrame"
        assert len(obscal)==len(model.inputs), \
                    "expected len(obscal)==len(model.inputs)"
        nvalid = obscal.notnull().sum(axis=1)
        nobs = obscal.shape[1]
        # .. this is not required in theory. To be removed perhaps
        # .. covariance matrix will discard missing data
        assert nvalid.isin([0, nobs]).all(), "Missing values should "+\
                                    "be identical across obs variables."
        self.obscal = obscal
        self._obs = obscal.copy()
        self.obs_variables = obscal.columns.tolist()
        mv = model.outputs_names
        assert all([v in mv for v in self.obs_variables]), "expected all obs "+\
                    "variables to be in the list of model outputs"

        # Add obs variables to assimilated states
        self.assim_states = list(set(assim_states + self.obs_variables))

        # inputs
        time = obscal.index
        self.rain = pd.Series(model.inputs[:, 0], index=time)
        self.train = self.transP.forward(self.rain)

        self.evap = pd.Series(model.inputs[:, 1], index=time)
        self.tevap = self.transE.forward(self.evap)

        self.maxrain_offset = 10
        self.maxrain_mult = 3

        # Model
        nvalo = model.outputs.shape[0]
        errmess = f"Expected model to be allocated with "+\
            f"len(outputs)={len(obscal)}, got {nvalo}."
        assert nvalo == len(obscal), errmess
        self.model = model

        # Perturbations + obs
        for n in set(self.perturb_variables+self.obs_variables):
            nn = n if n in self.perturb_variables else f"{n}_obs"
            assert nn in stdfacts, f"Expected {nn} in stdfacts"
            stdfacts[nn] = float(stdfacts[nn])
            assert n in rhofacts, f"Expected {nn} in rhofacts"
            rhofacts[nn] = float(rhofacts[nn])

        self.stdfacts = stdfacts
        self.rhofacts = rhofacts
        self.variabcorrfact = float(variabcorrfact)
        self.locdur = 1000 if ensmoother==1 else int(locdur)

        # Assimilated variable clipping
        self.clip = clip == 1

        # Debug config
        self.debug = debug == 1
        self.plot_freq = 30
        self.plot_dir = None
        self.plot_ax_size = (20, 4)
        self.plot_separate_figs = False
        self.plot_period = None

    @property
    def nstates_perturb(self):
        return len(self.perturb_variables)

    @property
    def nstates_assim(self):
        return len(self.assim_states)

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, val):
        assert isinstance(val, pd.DataFrame), "expected obs to be pd.DataFrame."
        assert len(val)==len(self.obscal), \
                    "expected len(obs)==len(obscal)."
        self._obs = val

    def initialise(self):
        nens = self.nens

        # defines min/max corrected rainfall
        transP, rain = self.transP, self.rain
        maxrain_offset, maxrain_mult = self.maxrain_offset, self.maxrain_mult
        tP_max = transP.forward(np.maximum(maxrain_offset, maxrain_mult*rain))
        self.tP_max = np.repeat(tP_max[:, None], nens, 1)

        transE, evap = self.transE, self.evap
        self.tE_min = self.transE.forward(0.)

        # defines min corrected streamflow
        self.tQ_min = self.transQ.forward(1e-3)

        # Initial run
        nval = len(rain)
        model = self.model

        # starting initial conditions prior to perturbation
        # .. set perturbation to 0.
        for inv, vn in enumerate(model.inputs_names):
            if re.search("delta$", vn):
                model.inputs[:, inv] = 0.

        # .. set simulation span
        model.istart = 0
        model.iend = nval-1
        # .. prior model run to compute variable std/rho/covar
        model.run()
        sims0 = model.to_dataframe().copy()
        self.sims0 = sims0

        # Compute statistics for model variables
        self.sigs, self.rhos = {}, {}
        transQ = self.transQ
        xpert = {}
        pvars = set(self.perturb_variables+["S", "R"])
        for v in pvars:
            xpert[v] = sims0.loc[:, v]

            if v in ["P", "P3"]:
                xpert[v] = transP.forward(xpert[v])
            elif v in ["E"]:
                xpert[v] = transE.forward(xpert[v])
            elif v in ["Q", "Q_obs"]:
                xpert[v] = transQ.forward(xpert[v])

            self.sigs[v], self.rhos[v] = compute_sig_and_rho(xpert[v], \
                                                self.stdfacts[v], \
                                                self.rhofacts[v])
        for v in self.obs_variables:
            vv = f"{v}_obs"
            xpert[vv] = self.obscal.loc[:, v]

            if v in ["P", "P3"]:
                xpert[vv] = transP.forward(xpert[vv])
            elif v in ["Q"]:
                xpert[vv] = transQ.forward(xpert[vv])

            self.sigs[vv], self.rhos[vv] = compute_sig_and_rho(xpert[vv], \
                                                self.stdfacts[vv], \
                                                self.rhofacts[vv])
        # Define variable correlations
        xdf = pd.DataFrame(xpert)
        self.variabcorr = compute_corr(xdf, self.variabcorrfact)

        # Find times with obs data for preceeding time step
        self.ifilter = np.where(self.obscal.notnull().all(axis=1).values)[0]+1

        # .. only the last time step if Ens Smoother.
        if self.ensmoother==1:
            self.ifilter = [self.ifilter[-1]]

        # Randomise obs independentaly from model errors
        varnames = [f"{n}_obs" for n in self.obs_variables]
        SigmaO, _ = get_sigma(varnames, self.sigs, self.rhos, \
                                    self.variabcorr, nval)
        oerr = sample(SigmaO, nens)

        # .. transform Q data
        iQo = self.obs_variables.index("Q")
        D0 = self.obscal.values.copy()
        D0[:, iQo] = transQ.forward(D0[:, iQo])
        D = D0.ravel()[:, None]+oerr

        self.D = D
        self.R = SigmaO

        # Sample perturbation from gaussian AR1 noise on state variables
        # i.e. transformed rainfall error, S and R
        SigmaE, status = get_sigma(self.perturb_variables, self.sigs, \
                                self.rhos, self.variabcorr, nval+1)
        self.SigmaE = SigmaE
        self.Xperturb = sample(SigmaE, nens)

        # Set parameters
        logparams0 = np.log(self.model.params.values)
        SigmaP = np.diag([self.stdfacts[n] for n in model.params.names])**2
        self.SigmaP = SigmaP
        perr = sample(SigmaP, nens)
        self.logparams = logparams0[:, None]+perr
        self.logparams0 = logparams0

        # initialise matrix containing analysed states.
        # For S and R, these are store values at the beginning of
        # the time step after perturbation (i.e. not the same as Sini and Rini)
        # This does not apply to P, P3 and Q.
        Xa = np.nan*np.zeros((nval*self.nstates_assim, nens))
        if self.assim_params == 1:
            # if we assimilate parameters, add prior for parameters (i.e. Xf)
            Xa = np.row_stack([self.logparams, Xa])

        self.Xa = Xa

        # Set initial conditions
        model.initialise_fromdata()
        Sini, Rini = model.states.values

        SigmaS, _ = get_sigma(["S"], self.sigs, self.rhos, self.variabcorr, 1)
        serr = sample(SigmaS, nens).squeeze()
        self.Sini = np.clip(Sini+serr, 0, model.X1)

        SigmaR, _ = get_sigma(["R"], self.sigs, self.rhos, self.variabcorr, 1)
        rerr = sample(SigmaR, nens).squeeze()
        self.Rini = np.clip(Rini+rerr, 0, model.Xr)


    def openloop(self, tstart, tend):
        nsim = tend-tstart
        nens = self.nens
        nstp = self.nstates_perturb
        nsta = self.nstates_assim
        perturb_variables = self.perturb_variables
        model = self.model
        transP = self.transP
        transE = self.transE
        transQ = self.transQ

        # Remove safety to go faster
        transP.check_input_arrays = False
        transQ.check_input_arrays = False

        # Instantiate open loop ensemble data
        X = np.zeros((nsta*nsim, nens))

        # Indexes of model output variables
        onames = model.outputs_names
        mP  = onames.index("P")
        mE  = onames.index("E")
        mAE  = onames.index("AE")
        mQ  = onames.index("Q")
        mP3 = onames.index("P3")
        mS  = onames.index("S")
        mR  = onames.index("R")
        mF  = onames.index("F")

        # Model has already been allocated
        # (i.e. inputs data + outputs set to nan beyon tend-1)
        model.istart = tstart
        model.iend = tend-1

        for iens in range(nens):
            # Set perturbations
            epert = self.Xperturb[nstp*tstart:nstp*tend, iens]
            epert = epert.reshape((nsim, nstp))
            idx = [model.inputs_names.index(f"{'' if n in ['S', 'R'] else 't'}{n}delta") \
                                for n in perturb_variables]
            model.inputs[tstart:tend, idx] = epert

            # Set parameters
            if self.assim_params == 1:
                model.params.values = np.exp(self.logparams[:, iens])

            # Initialise model
            xini = [self.Sini[iens], self.Rini[iens]]
            model.initialise(xini)

            # Run model over open loop period
            model.run()

            # Store asssimilated states simulations
            # .. S, R at the BEGINNING OF TIMESTEP !!!!
            # .. (this a causal assimilation variable)
            s = np.zeros((nsim, nsta))
            moutputs = model.outputs
            for istate, sname in enumerate(self.assim_states):
                if sname == "S":
                    sim = np.concatenate([[xini[0]], \
                                        moutputs[tstart:tend-1, mS]])
                    s[:, istate] = sim

                elif sname == "R":
                    sim = np.concatenate([[xini[1]], \
                                    moutputs[tstart:tend-1, mR]])
                    s[:, istate] = sim

                elif sname == "P":
                    P = np.ascontiguousarray(moutputs[tstart:tend, [mP]])
                    # Here we need to transform P because the assimilated
                    # states is transformed rainfall, not raw rainfall
                    s[:, istate] = transP.forward(P)

                elif sname == "E":
                    E = np.ascontiguousarray(moutputs[tstart:tend, [mE]])
                    # Here we need to transform E because the assimilated
                    # states is transformed evap, not raw evap
                    s[:, istate] = transE.forward(E)

                elif sname == "AE":
                    AE = np.ascontiguousarray(moutputs[tstart:tend, [mAE]])
                    # Use transE to transform AE
                    s[:, istate] = transE.forward(AE)

                elif sname == "P3":
                    P3 = np.ascontiguousarray(moutputs[tstart:tend, [mP3]])
                    # Here we need to transform P3 because the assimilated
                    # states is transformed effective rainfall, not
                    # raw effective rainfall
                    s[:, istate] = transP.forward(P3)

                elif sname == "F":
                    F = moutputs[tstart:tend, mF]
                    # no transformation of F
                    s[:, istate] = F

                elif sname == "Q":
                    Q = np.ascontiguousarray(moutputs[tstart:tend, [mQ]])
                    # Here we need to transform Q because the assimilated
                    # states is transformed runoff, not raw runoff
                    s[:, istate] = transQ.forward(Q)

                else:
                    raise ValueError("Do not know how to generate open loop"+\
                                f" for state {state}.")

            X[:, iens] = s.ravel().copy()

            # update stores for next run of open loop
            # we run the model over the last time step to obtain
            # the store level at the end of the period prior to
            # pertubation

            # .. production store
            Sprev = model.outputs[tend-1, mS]
            P = model.outputs[tend-1, mP] # Perturbed rainfall
            E = model.outputs[tend-1, mE] # Perturbed PET
            Sini = c_pydaisi.gr2m_S_fun(self.X1, Sprev, P, E)
            P3 = c_pydaisi.gr2m_P3_fun(self.X1, Sprev, P, E)
            self.Sini[iens] = Sini

            # .. routing store
            Rprev = model.outputs[tend-1, mR]
            self.Rini[iens] = c_pydaisi.gr2m_R_fun(self.X2, self.Xr, Rprev, P3)

        # Put back safety
        transP.check_input_arrays = True
        transQ.check_input_arrays = True

        # Set back original parameters
        self.model.params.values = np.exp(self.logparams0)

        return X


    def analysis(self, tend, Xf, HXf):
        if not hasattr(self, "model"):
            raise ValueError("EnKS is not initialised")

        nstates = self.nstates_assim
        nobs = len(self.obs_variables)
        nens = self.nens
        nparams = self.nparams

        # Ensemble stats
        EXf = np.mean(Xf[:nparams+nstates*tend], axis=1)
        EHXf = np.mean(HXf, axis=1)
        # .. Ensemble deviations
        Xprime = Xf[:nparams+nstates*tend]-EXf[:, None]
        HXprime = HXf-EHXf[:, None]
        # .. Covariance matrices
        HCH = HXprime.dot(HXprime.T)/(nens-1)
        CH = Xprime.dot(HXprime.T)/(nens-1)

        # Gain obs matrices up to time=t
        W = self.D[:nobs*tend, :]-HXf
        P = HCH+self.R[:nobs*tend][:, :nobs*tend]
        np.linalg.cholesky(P)

        # Identify non-missing values from first ensemble
        iok = ~np.isnan(W[:, 0])

        ## EnKS analysis for non-missing values
        ## .. update = CH x (HCH+R)^-1 x (D-HXf)
        V = solve(P[iok][:, iok], W[iok], assume_a="pos")
        update = CH[:, iok].dot(V)

        # reduction of spurious time correlation
        A = (np.arange(tend)<self.locdur).astype(float)[::-1]
        A = np.row_stack([np.ones(nparams)[:, None], \
                        np.repeat(A[:, None], nstates, 0)])
        update *= A

        # .. Apply EnKS update
        xtmp = Xf+update

        if self.clip == 1:
            # .. clip variables outside feasible domain
            # .... P < Pmax
            states_names = self.assim_states
            if "P" in states_names:
                iP = states_names.index("P")
                tP = np.minimum(xtmp[nparams+iP::nstates, :], self.tP_max[:tend])
                xtmp[nparams+iP::nstates, :] = tP

            # ... E>=0
            if "E" in states_names:
                iE = states_names.index("E")
                tE = np.maximum(xtmp[nparams+iE::nstates, :], self.tE_min)
                xtmp[nparams+iE::nstates, :] = tE

            # ... AE>=0
            if "AE" in states_names:
                iAE = states_names.index("AE")
                tAE = np.maximum(xtmp[nparams+iAE::nstates, :], self.tE_min)
                xtmp[nparams+iAE::nstates, :] = tAE

            # .... S in [0, X1]
            if "S" in states_names:
                iS = states_names.index("S")
                if self.assim_params == 1:
                    iX1 = 0
                    X1s = np.repeat(np.exp(xtmp[iX1, :])[None, :], tend, axis=0)
                else:
                    X1s = self.X1
                xtmp[nparams+iS::nstates, :] = \
                                np.clip(xtmp[nparams+iS::nstates, :], 0, X1s)
            # .... P3 < P
            if "P3" in states_names:
                iP3 = states_names.index("P3")
                tP3 = np.minimum(xtmp[nparams+iP3::nstates, :], self.tP_max[:tend])
                xtmp[nparams+iP3::nstates, :] = tP3
            # .... R in [0, Xr]
            if "R" in states_names:
                iR = states_names.index("R")
                xtmp[nparams+iR::nstates, :] = \
                                np.clip(xtmp[nparams+iR::nstates, :], 0, self.Xr)

            # Don't do that otherwise the variance becomes close to 0.
            # and it messes up analysis
            #if "Q" in states_names:
            #    iQ = states_names.index("Q")
            #    xtmp[nparams+iQ::nstates, :] = \
            #                    np.maximum(xtmp[nparams+iQ::nstates, :], self.tQ_min)

        # .. update Xa BY REF !!!!
        self.Xa[:nparams+nstates*tend, :] = xtmp


    def run(self, context="EnKS_run", message="Running Enks"):
        tstart = 0
        nstates = self.nstates_assim
        nparams = self.nparams
        assim_params = self.assim_params
        idx = self.ifilter
        tbar = tqdm(enumerate(idx), desc=message, total=len(idx), \
                        disable=not self.debug)

        # set outputs to nan to make sure open loop overwrites them
        self.model.outputs = np.nan*self.model.outputs

        # Run smoother
        for itime, tend in tbar:
            # Run open loop
            X = self.openloop(tstart, tend)

            # Concatenate prior analysis (t<tstart) and openloop
            # (t_start <= t < t_end)
            Xf = self.Xa[:nparams+nstates*tend, :].copy()
            Xf[nparams+nstates*tstart:nparams+nstates*tend, :] = X

            # Get analysed output (i.e. assimilated variables)
            iO = [self.assim_states.index(v) \
                        for v in self.obs_variables]
            ixf = nparams+nstates*np.arange(tend)[:, None]+np.array(iO)[None, :]
            ixf = ixf.ravel()
            HXf = Xf[ixf, :]

            # Run EnKS and update Xa
            self.analysis(tend, Xf, HXf)

            # Loop
            tstart = tend

            # Plot
            if self.debug and (tend%self.plot_freq == 0 or itime==len(idx)-1):
                f = self.plot_dir
                if f is None:
                    errmess = f"plot_dir is None"
                    raise ValueError(errmess)

                fname = self.plot_dir/f"enks_{context}_T{tend:03d}.png"
                self.plot(fname)


    def plot(self, fname):
        # Get data
        nstates = self.nstates_assim
        nparams = self.nparams
        transP, transE, transQ = self.transP, self.transE, self.transQ
        sims0, rain, evap, obs, obscal = self.sims0, self.rain, \
                                self.evap, self.obs, self.obscal
        Xa = self.Xa
        time = obscal.index

        awidth, aheight  = self.plot_ax_size

        # Plot params if assimilating params
        if nparams>0:
            plt.close("all")
            prior = np.exp(self.logparams)
            post = np.exp(Xa[:nparams, :])

            figsize = (awidth/2, aheight*nparams)
            fig = plt.figure(figsize=figsize, layout="constrained")
            pnames = self.model.params.names.tolist()
            mosaic = [[pn] for pn in pnames]
            axs = fig.subplot_mosaic(mosaic)
            for aname, ax in axs.items():
                ip = pnames.index(aname)
                ax.hist(prior[ip, :], bins=30, color="grey", alpha=0.3, \
                                    label="GR2M prior")
                ax.hist(post[ip, :], bins=30, color="tab:orange", alpha=0.3, \
                                    label="DA")
                ax.legend(loc=2)
                ax.text(0.98, 0.98, aname, fontweight="bold", \
                            transform=ax.transAxes, \
                            ha="right", va="top")

            fp = fname.parent / f"params_{fname.stem}.png"
            fig.savefig(fp)

        plot_period = self.plot_period
        if not plot_period is None:
            ystart, yend = plot_period
            tsel = (time.year >= ystart) & (time.year <= yend)
        else:
            tsel = np.ones(len(time)).astype(bool)

        plt.close("all")
        sepfigs = self.plot_separate_figs
        if not sepfigs:
            figsize = (awidth, aheight*nstates)
            try:
                fig = plt.figure(figsize=figsize, layout="constrained")
            except:
                # Matplotlib problem in Azure devops
                fig = plt.figure(figsize=figsize)

            mosaic = [[vn] for vn in self.assim_states]
            axs = fig.subplot_mosaic(mosaic)

        for aname in self.assim_states:
            if sepfigs:
                fig, ax = plt.subplots(figsize=(awidth, aheight), \
                                            layout="constrained")
            else:
                ax = axs[aname]

            # Plot assimilated states
            iax = self.assim_states.index(aname)
            ts = Xa[nparams+iax::nstates]

            if aname in ["P", "P3"]:
                s = pd.DataFrame(transP.backward(ts), index=time)
            elif aname in ["E", "AE"]:
                s = pd.DataFrame(transE.backward(ts), index=time)
            elif aname == "Q":
                s = pd.DataFrame(transQ.backward(ts), index=time)
            else:
                s = pd.DataFrame(ts, index=time)

            # .. quantiles
            sq = s.quantile([0.5, 0.05, 0.95], axis=1).T
            #t = date2num(sq.index)
            ax.fill_between(sq.index[tsel], sq.iloc[tsel, 1].values, \
                                sq.iloc[tsel, 2].values, \
                                facecolor="tab:orange", alpha=0.2, \
                                edgecolor="none",\
                                label=f"{aname} XA 90% CI")

            smean = s.mean(axis=1)
            lab = f"{aname} XA-Mean"
            ax.plot(smean.index, smean.values, label=lab, color="tab:orange", lw=2)

            # Plot additional data
            if aname == "S":
                pd.Series(sims0.S.shift(1).values[tsel], index=time[tsel]).plot(ax=ax, \
                            color="green", lw=2, label="S GR2M")
            elif aname == "R":
                pd.Series(sims0.R.shift(1).values[tsel], index=time[tsel]).plot(ax=ax, \
                            color="green", lw=2, label="R GR2M")
            elif aname == "P3":
                pd.Series(sims0.P3.values[tsel], index=time[tsel]).plot(ax=ax, \
                            color="green", lw=2, label="P3 GR2M")
            elif aname == "AE":
                pd.Series(sims0.AE.values[tsel], index=time[tsel]).plot(ax=ax, \
                            color="green", lw=2, label="AE GR2M")
            elif aname == "F":
                pd.Series(sims0.F.values[tsel], index=time[tsel]).plot(ax=ax, \
                            color="green", lw=2, label="F GR2M")
            elif aname == "P":
                pd.Series(rain[tsel], index=time[tsel]).plot(ax=ax, color="green", \
                                lw=2, label="Rain Obs")
            elif aname == "E":
                pd.Series(evap[tsel], index=time).plot(ax=ax, color="green", \
                                lw=2, label="PET Obs")
            elif aname == "Q":
                pd.Series(sims0.Q.values[tsel], index=time[tsel]).plot(ax=ax, \
                                            color="green", \
                                            lw=1, label="sim")

                obs.loc[tsel, "Q"].plot(ax=ax, color="tab:red", alpha=0.9, \
                                            lw=2, label="Q Obs")

                obscal.loc[tsel, "Q"].plot(ax=ax, marker="+", markersize=8,\
                                label="Q Obs Cal", \
                                color="none", markeredgecolor="tab:red")

                y1 = obs.loc[tsel, "Q"].max()*2
                ax.set_ylim((0, y1))
                fwd = lambda x: np.sqrt(x+1)
                inv = lambda y: y*y-1
                ax.set_yscale("function", functions=(fwd, inv))
                ax.set_ylim((0, y1))

            ax.legend(loc=1, fontsize="large")
            tt = time[tsel]
            ax.set_xlim((tt[0], tt[-1]))
            ax.text(0.02, 0.98, aname, transform=ax.transAxes, \
                        fontweight="bold", ha="left", va="top")


            if sepfigs:
                fname_state = fname.parent / f"{fname.stem}_{aname}.png"
                fig.savefig(fname_state)

        if not sepfigs:
            fig.savefig(fname)


