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


def get_sigma(varnames, sigs, variabcorr, nval):
    nvar = len(varnames)
    assert all([(n in sigs) for n in varnames]), \
                        "Expected all varnames in sigs"

    # Build covariance matrix
    Sigma = np.zeros((nval*nvar, nval*nvar))
    indices = np.arange(0, nval*nvar, nvar)
    eye = np.eye(nval)
    for i, j in combwr(range(nvar), 2):
        vi, vj = varnames[i], varnames[j]
        si, sj = sigs[vi], sigs[vj]
        ii = (indices+i)[:, None]
        jj = (indices+j)[None, :]
        cij = variabcorr.loc[vi, vj]

        S = si*sj*cij*eye
        Sigma[ii, jj] = S
        Sigma[jj.T, ii.T] = S.T

    return Sigma


def sample(Sigma, nens):
    eig, Q = np.linalg.eig(Sigma)
    U = np.random.normal(size=(Sigma.shape[0], nens))
    return Q.dot(np.diag(np.sqrt(eig))).dot(U)


def compute_sig(x, stdfact):
    assert stdfact>=1e-6 and stdfact<=1.
    sig = np.nanstd(x)*stdfact # ensures sig>1e-6
    sig = 1e-3 if np.isnan(sig) or sig<1e-3 else sig
    return sig



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




class EnSmooth():
    def __init__(self, model, \
                    obscal, \
                    stdfacts, \
                    debug, nens=200):
        # Dims
        self.nens = nens

        # List of perturbed states
        self.perturb_variables = ["P", "E", "P3", "S", "R", "Q"]

        # Assimilated states
        # X0  assimilate production states
        # X1  assimilate routing states
        # X2  assimilate both
        assim_states = ["S", "P3", "R", "Q"]

        # GR2M params
        self.X1 = model.X1
        self.Xr = model.Xr
        self.X2 = model.X2

        # Transforms
        xclip = -model.nu+2e-4
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
        to_be_added = list(set(self.obs_variables)-set(assim_states))
        self.assim_states = assim_states + to_be_added

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

        self.stdfacts = stdfacts

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
        self.sigs = {}
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

            self.sigs[v] = compute_sig(xpert[v], self.stdfacts[v])

        for v in self.obs_variables:
            vv = f"{v}_obs"
            xpert[vv] = self.obscal.loc[:, v]

            if v in ["P", "P3"]:
                xpert[vv] = transP.forward(xpert[vv])
            elif v in ["Q"]:
                xpert[vv] = transQ.forward(xpert[vv])

            self.sigs[vv] = compute_sig(xpert[vv], self.stdfacts[vv])

        # Variable correlations
        xdf = pd.DataFrame(xpert)
        self.variabcorr = compute_corr(xdf, 1.0)

        # Find times with obs data for preceeding time step
        self.ifilter = np.where(self.obscal.notnull().all(axis=1).values)[0]+1
        # .. only the last time step if Ens Smoother.
        self.ifilter = [self.ifilter[-1]]

        # Randomise obs independentaly from model errors
        varnames = [f"{n}_obs" for n in self.obs_variables]
        SigmaO = get_sigma(varnames, self.sigs, \
                                        self.variabcorr, nval)
        oerr = sample(SigmaO, nens)

        # .. transform Q data
        iQo = self.obs_variables.index("Q")
        D0 = self.obscal.values.copy()
        D0[:, iQo] = transQ.forward(D0[:, iQo])
        D = D0.ravel()[:, None]+oerr

        self.D = D
        self.R = SigmaO

        # Sample perturbation from gaussian noise in transformed space
        SigmaE = get_sigma(self.perturb_variables, self.sigs, \
                                    self.variabcorr, nval+1)
        self.SigmaE = SigmaE
        self.Xperturb = sample(SigmaE, nens)

        # initialise matrix containing analysed states.
        # For S and R, these are store values at the beginning of
        # the time step after perturbation (i.e. not the same as Sini and Rini)
        # This does not apply to P, P3 and Q.
        self.Xa = np.nan*np.zeros((nval*self.nstates_assim, nens))

        # Set initial conditions with perturbation
        model.initialise_fromdata()
        Sini, Rini = model.states.values

        SigmaS = get_sigma(["S"], self.sigs, self.variabcorr, 1)
        serr = sample(SigmaS, nens).squeeze()
        self.Sini = np.clip(Sini+serr, 0, model.X1)

        SigmaR = get_sigma(["R"], self.sigs, self.variabcorr, 1)
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
            idx = [model.inputs_names.index(\
                            f"{'' if n in ['S', 'R'] else 't'}{n}delta") \
                                for n in perturb_variables]
            model.inputs[tstart:tend, idx] = epert

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

        # Put back safety in transform functions
        transP.check_input_arrays = True
        transQ.check_input_arrays = True

        return X


    def analysis(self, tend, Xf, HXf):
        if not hasattr(self, "model"):
            raise ValueError("EnSmooth is not initialised")

        nstates = self.nstates_assim
        nobs = len(self.obs_variables)
        nens = self.nens
        nparams = 0 # No assimilation of parameters

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

        ## EnSmooth analysis for non-missing values
        ## .. update = CH x (HCH+R)^-1 x (D-HXf)
        V = solve(P[iok][:, iok], W[iok], assume_a="pos")
        update = CH[:, iok].dot(V)

        # .. Apply EnSmooth update
        xtmp = Xf+update
        self.Xa[:nparams+nstates*tend, :] = xtmp


    def run(self, context="EnSmooth_run", message="Running Enks"):
        tstart = 0
        nstates = self.nstates_assim
        nparams = 0 # No assimilation of parameters
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

            # Run EnSmooth and update Xa
            self.analysis(tend, Xf, HXf)

            # Loop
            tstart = tend

            # Plot
            if self.debug and (tend%self.plot_freq == 0 or itime==len(idx)-1):
                f = self.plot_dir
                if f is None:
                    errmess = f"plot_dir is None"
                    raise ValueError(errmess)

                fname = self.plot_dir/f"ensmooth_{context}_T{tend:03d}.png"
                self.plot(fname)


    def plot(self, fname):
        # Get data
        nstates = self.nstates_assim
        nparams = 0
        transP, transE, transQ = self.transP, self.transE, self.transQ
        sims0, rain, evap, obs, obscal = self.sims0, self.rain, \
                                self.evap, self.obs, self.obscal
        Xa = self.Xa
        time = obscal.index

        awidth, aheight  = self.plot_ax_size

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

                obs.loc[tsel, "Q"].plot(ax=ax, color="k", linestyle="--", alpha=0.9, \
                                            lw=2, label="Q Obs")

                obscal.loc[tsel, "Q"].plot(ax=ax, marker="o", markersize=8,\
                                label="Q Obs Cal", \
                                color="k")

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


