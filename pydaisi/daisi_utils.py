import re
import warnings
from itertools import product as prod
import math
import numpy as np
import pandas as pd

import warnings

from scipy.stats import theilslopes
from scipy.stats import invgamma, norm

HAS_STATSMODELS = False
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    pass

from hydrodiy.stat import sutils

import c_pydaisi

def get_varname(varname):
    if varname == "S":
        return r"$s^+$"
    elif varname == "P3":
        return r"$p_e$"
    elif varname == "R":
        return r"$r^+$"
    elif varname == "Q":
        return r"$q$"
    elif varname == "P":
        return "Rain"
    elif varname == "E":
        return "PET"
    else:
        mess = f"Cannot find {state}"
        raise ValueError(mess)

def get_metricname(metric, long=False):
    n = re.sub("^.*_|-.*$", "", metric)
    if n == "ABSFDCFIT100":
        nm = r"$F_B$"
        lnm = f"Flow duration curve bias ({nm})"

    elif n == "ELASTrelRAIN":
        nm = r"$\epsilon_P$"
        lnm = f"Elasticity to rainfall ({nm})"

    elif n == "ELASTrelEVAP":
        nm = r"$\epsilon_E$"
        lnm = f"Elasticity to PET ({nm})"

    elif n == "NSELOG":
        nm = r"$NSE_{log}$"
        lnm = f"NSE on log flows ({nm})"

    elif n == "NSERECIP":
        nm = r"$NSE_{rec}$"
        lnm = f"NSE on reciprocal flows ({nm})"

    elif n == "SPLITKGE":
        nm = r"$KGE_{split}$"
        lnm = f"Split KGE ({nm})"

    elif n.startswith("PMR"):
        ny = int(re.sub("PMR|Y", "", n))
        nm = r"$PMR_{{{ny}}}$".format(ny=ny)
        lnm = f"PMR {ny} years ({nm})"

    elif n == "ABSBIAS":
        nm = r"$1-|B|$"
        lnm = f"Absolute bias index ({nm})"
    else:
        nm = f"${n}$"
        lnm = nm

    return lnm if long else nm


# --- Reference GR2M functions ---
# .. normalisation
def to1d(x):
    x = np.array(x).astype(np.float64)
    assert x.ndim == 1
    out = np.zeros_like(x)
    return x, out

def gr2m_prod_S_raw2norm(X1, S):
    S, out = to1d(S)
    c_pydaisi.vect_gr2m_prod_S_raw2norm(X1, S, out)
    return out

def gr2m_prod_S_norm2raw(X1, u):
    u, out = to1d(u)
    c_pydaisi.vect_gr2m_prod_S_norm2raw(X1, u, out)
    return out

def gr2m_prod_P_raw2norm(X1, P):
    P, out = to1d(P)
    c_pydaisi.vect_gr2m_prod_P_raw2norm(X1, P, out)
    return out

def gr2m_prod_P_norm2raw(X1, pn):
    pn, out = to1d(pn)
    c_pydaisi.vect_gr2m_prod_P_norm2raw(X1, pn, out)
    return out

def gr2m_prod_E_raw2norm(X1, E):
    E, out = to1d(E)
    c_pydaisi.vect_gr2m_prod_E_raw2norm(X1, E, out)
    return out

def gr2m_prod_E_norm2raw(X1, en):
    en, out = to1d(en)
    c_pydaisi.vect_gr2m_prod_E_norm2raw(X1, en, out)
    return out

def gr2m_prod_AE_raw2norm(X1, AE):
    AE, out = to1d(AE)
    c_pydaisi.vect_gr2m_prod_AE_raw2norm(X1, AE, out)
    return out

def gr2m_prod_AE_norm2raw(X1, aen):
    aen, out = to1d(aen)
    c_pydaisi.vect_gr2m_prod_AE_norm2raw(X1, aen, out)
    return out

def gr2m_prod_P3_raw2norm(X1, P3):
    P3, out = to1d(P3)
    c_pydaisi.vect_gr2m_prod_P3_raw2norm(X1, P3, out)
    return out

def gr2m_prod_P3_norm2raw(X1, p3n):
    p3n, out = to1d(p3n)
    c_pydaisi.vect_gr2m_prod_P3_norm2raw(X1, p3n, out)
    return out

def gr2m_rout_P3_raw2norm(X2, Xr, P3):
    P3, out = to1d(P3)
    c_pydaisi.vect_gr2m_rout_P3_raw2norm(X2, Xr, P3, out)
    return out

def gr2m_rout_P3_norm2raw(X2, Xr, p3n):
    p3n, out = to1d(p3n)
    c_pydaisi.vect_gr2m_rout_P3_norm2raw(X2, Xr, p3n, out)
    return out

def gr2m_rout_Rstart_raw2norm(X2, Xr, R):
    R, out = to1d(R)
    c_pydaisi.vect_gr2m_rout_Rstart_raw2norm(X2, Xr, R, out)
    return out

def gr2m_rout_Rstart_norm2raw(X2, Xr, v):
    v, out = to1d(v)
    c_pydaisi.vect_gr2m_rout_Rstart_norm2raw(X2, Xr, v, out)
    return out

def gr2m_rout_Rend_raw2norm(X2, Xr, R):
    R, out = to1d(R)
    c_pydaisi.vect_gr2m_rout_Rend_raw2norm(X2, Xr, R, out)
    return out

def gr2m_rout_Rend_norm2raw(X2, Xr, v):
    v, out = to1d(v)
    c_pydaisi.vect_gr2m_rout_Rend_norm2raw(X2, Xr, v, out)
    return out

def gr2m_rout_F_raw2norm(X2, Xr, F):
    F, out = to1d(F)
    c_pydaisi.vect_gr2m_rout_F_raw2norm(X2, Xr, F, out)
    return out

def gr2m_rout_F_norm2raw(X2, Xr, fn):
    fn, out = to1d(fn)
    c_pydaisi.vect_gr2m_rout_F_norm2raw(X2, Xr, fn, out)
    return out

def gr2m_rout_Q_raw2norm(X2, Xr, Q):
    Q, out = to1d(Q)
    c_pydaisi.vect_gr2m_rout_Q_raw2norm(X2, Xr, Q, out)
    return out

def gr2m_rout_Q_norm2raw(X2, Xr, qn):
    qn, out = to1d(qn)
    c_pydaisi.vect_gr2m_rout_Q_norm2raw(X2, Xr, qn, out)
    return out


# .. time step forward
def gr2m_S_fun(X1, S, P, E):
    S, P, E = np.array(S), np.array(P), np.array(E)
    assert S.ndim == 1
    assert P.ndim == 1
    assert E.ndim == 1
    out = np.zeros_like(S)
    c_pydaisi.vect_gr2m_S_fun(X1, S, P, E, out)
    return out

def gr2m_S_fun_normalised(X1, u, phi, psi):
    Sstart = gr2m_prod_S_norm2raw(X1, u)
    P = gr2m_prod_P_norm2raw(X1, phi)
    E = gr2m_prod_E_norm2raw(X1, psi)
    Send = gr2m_S_fun(X1, Sstart, P, E)
    return gr2m_prod_S_raw2norm(X1, Send)



def gr2m_P3_fun(X1, S, P, E):
    S, P, E = np.array(S), np.array(P), np.array(E)
    assert S.ndim == 1
    assert P.ndim == 1
    assert E.ndim == 1
    out = np.zeros_like(S)
    c_pydaisi.vect_gr2m_P3_fun(X1, S, P, E, out)
    return out

def gr2m_P3_fun_normalised(X1, u, phi, psi):
    Sstart = gr2m_prod_S_norm2raw(X1, u)
    P = gr2m_prod_P_norm2raw(X1, phi)
    E = gr2m_prod_E_norm2raw(X1, psi)
    P3 = gr2m_P3_fun(X1, Sstart, P, E)
    return gr2m_prod_P3_raw2norm(X1, P3)


def gr2m_AE_fun(X1, S, P, E):
    P3 = gr2m_P3_fun(X1, S, P, E)
    Send = gr2m_S_fun(X1, S, P, E)
    return S-Send+P-P3

def gr2m_AE_fun_normalised(X1, u, phi, psi):
    Sstart = gr2m_prod_S_norm2raw(X1, u)
    P = gr2m_prod_P_norm2raw(X1, phi)
    E = gr2m_prod_E_norm2raw(X1, psi)
    AE = gr2m_AE_fun(X1, Sstart, P, E)
    return gr2m_prod_AE_raw2norm(X1, AE)



def gr2m_R_fun(X2, Xr, R, P3):
    R, P3 = np.array(R), np.array(P3)
    assert R.ndim == 1
    assert P3.ndim == 1
    out = np.zeros_like(R)
    c_pydaisi.vect_gr2m_R_fun(X2, Xr, R, P3, out)
    return out

def gr2m_R_fun_normalised(X2, Xr, v, p3n):
    Rstart = gr2m_rout_Rstart_norm2raw(X2, Xr, v)
    P3 = gr2m_rout_P3_norm2raw(X2, Xr, p3n)
    Rend = gr2m_R_fun(X2, Xr, Rstart, P3)
    return gr2m_rout_Rend_raw2norm(X2, Xr, Rend)



def gr2m_Q_fun(X2, Xr, R, P3):
    R, P3 = np.array(R), np.array(P3)
    assert R.ndim == 1
    assert P3.ndim == 1
    assert R.ndim == 1
    out = np.zeros_like(R)
    c_pydaisi.vect_gr2m_Q_fun(X2, Xr, R, P3, out)
    return out

def gr2m_Q_fun_normalised(X2, Xr, v, p3n):
    Rstart = gr2m_rout_Rstart_norm2raw(X2, Xr, v)
    P3 = gr2m_rout_P3_norm2raw(X2, Xr, p3n)
    Q = gr2m_Q_fun(X2, Xr, Rstart, P3)
    return gr2m_rout_Q_raw2norm(X2, Xr, Q)


def gr2m_F_fun(X2, Xr, R, P3):
    Q = gr2m_Q_fun(X2, Xr, R, P3)
    Rend = gr2m_R_fun(X2, Xr, R, P3)
    return Rend-R-P3+Q

def gr2m_F_fun_normalised(X2, Xr, v, p3n):
    Rstart = gr2m_rout_Rstart_norm2raw(X2, Xr, v)
    P3 = gr2m_rout_P3_norm2raw(X2, Xr, p3n)
    F = gr2m_F_fun(X2, Xr, Rstart, P3)
    return gr2m_rout_F_raw2norm(X2, Xr, F)


# Utils functions
def toarray(x):
    # Pandas -> numpy
    if hasattr(x, "values"):
        x = x.values

    # Numpy -> Numpy 2d
    if hasattr(x, "ndim"):
        if x.ndim==0:
            x = x[None, None]
        elif x.ndim==1:
            x = x[:, None]
    else:
        x = np.array([[x]])

    if x.dtype != np.float64:
        x = x.astype(np.float64)

    if not x.data.contiguous:
        x = np.ascontiguousarray(x)

    return x


class Transform():
    """ Fast Box-Cox transform class """
    def __init__(self, lam, nu=0.1, xclip=None, check_input_arrays=True):
        self._lam = np.float64(lam)
        self._nu = np.float64(nu)
        if xclip is None:
            self._xclip = -self.nu+1e-4
        else:
            xclip = np.float64(xclip)
            assert xclip>-self.nu+1e-4, "xclip is too close to nu"
            self._xclip = xclip

        self.check_input_arrays=True

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, val):
        self._lam = np.float64(val)

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, val):
        self._nu = np.float64(val)

    @property
    def xclip(self):
        return self._xclip

    @xclip.setter
    def xclip(self, val):
        self._xclip = np.float64(val)

    def forward(self, x):
        lam, nu = self.lam, self.nu
        if self.check_input_arrays:
            x = toarray(x)
        y = np.zeros_like(x, dtype=np.float64)
        c_pydaisi.boxcox_forward_vect(lam, nu, x, y)
        return y.squeeze()

    def backward(self, y):
        lam, nu = self.lam, self.nu
        if self.check_input_arrays:
            y = toarray(y)
        x = np.zeros_like(y, dtype=np.float64)
        c_pydaisi.boxcox_backward_vect(lam, nu, y, x)
        return x.squeeze()

    def perturb(self, x, eps):
        lam, nu = self.lam, self.nu
        if self.check_input_arrays:
            x = toarray(x)
            eps = toarray(eps)
        y = np.zeros_like(eps, dtype=np.float64)

        c_pydaisi.boxcox_perturb_vect(lam, nu, self.xclip, x, eps, y)
        return y.squeeze()



class BayesianRegression():
    def __init__(self, X, y, rcond=1e-6):
        # Check inputs
        self.rcond = float(rcond)
        assert self.rcond>=0 and self.rcond<0.5
        self.nval, self.nparams = X.shape
        errmsg = "Error in inputs dimensions"
        assert y.shape[0] == self.nval, errmsg

        errmsg = "No nan allowed in X or y"
        assert np.all(~np.isnan(X)), errmsg
        assert np.all(~np.isnan(y)), errmsg

        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            names = X.columns.tolist()
        else:
            names = [f"beta{i}" for i in range(1, X.shape[1]+1)]
        X = np.array(X)

        # Variables
        self.X = X
        self.names = names
        self.y = y
        self.has_prior = False

        # SVD using truncated matrices
        # U[nxk], S[k], V[kxk]
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            wmess = f"[BayesianRegression] Added random noise to "+\
                    "predictor matrix X to avoid lack of SVD convergence."
            warnings.warn(wmess)
            e = 1e-10*np.random.uniform(-1, 1, X.shape)
            U, S, Vt = np.linalg.svd(X+e, full_matrices=False)

        self.U = U
        self.V = Vt.T
        self.S = S

        # no prior set up (i.e. uninformative)
        self.has_inf_prior = False
        self.eta0 = np.zeros(self.nparams)


    def solve(self):
        y = self.y
        S = self.S
        nval, nparams = self.nval, self.nparams
        eta0 = self.eta0

        # Uninformative prior
        # See Gelman, section 14.2 page 355
        # + need for elimation of very poorly conditions
        highsv = S>S.max()*self.rcond
        self.highsv = highsv
        Up = self.U[:, highsv]
        Sinv = (1/S[highsv])[:, None]
        etahat = (Sinv*Up.T).dot(y)

        yhat = (Up*(S[highsv][None, :])).dot(etahat)
        err = y-yhat

        ndegf = nval-highsv.sum()
        sigsqhat = np.sum(err*err)/ndegf

        # .. careful, the number of parameters is reduced here!
        an = ndegf/2
        bn = sigsqhat*an
        etan = eta0.copy()
        etan[highsv] = etahat
        # .. use SVD to compute square root of inverse
        M = Sinv[:, 0]

        ## Store
        self.yhat = yhat
        self.err = err
        self.etahat = etahat
        self.etan = etan
        self.betan = self.V.dot(etan)
        self.an = an
        self.bn = bn
        self.M = M


    def sample(self, nsamples):
        # Retrieve data
        betan, an, bn, M, V = self.betan, self.an, self.bn, self.M, self.V

        # Sample sigma
        sigs = np.sqrt(invgamma.rvs(a=an, scale=bn, size=nsamples))

        # Sample eta in transform space
        us = norm.rvs(size=(nsamples, len(M)))*M[None, :]
        if self.has_inf_prior:
            u = us
        else:
            # Operate on reduced variables and set
            # null space to 0.
            u = np.zeros((nsamples, self.nparams))
            u[:, self.highsv] = us
            lsv = ~self.highsv
            u[:, lsv] = self.eta0[None, lsv]

        # Back transform and add to betan
        betas = betan[None, :]+(sigs[:, None]*u).dot(V.T)

        return pd.DataFrame(betas, columns=self.names), \
                    pd.Series(sigs)


    def predict(self, betas, sigs, X=None):
        # Check inputs
        betas = np.array(betas)
        sigs = np.array(sigs)
        errmsg = "Wrong dimensions"
        assert betas.ndim == 2, errmsg
        assert sigs.ndim == 1, errmsg
        assert betas.shape[0] == sigs.shape[0], errmsg

        if X is None:
            X = self.X
        assert X.shape[1] == betas.shape[1], errmsg

        # Predict
        y1 = X.dot(betas.T)
        err = sigs[None, :]*np.random.normal(size=y1.shape)
        return y1+err


def model_elasticity(model, iactive, relative=False, ratios=[5, 10, 15]):
    inputs = model.inputs.copy()
    ratios = np.sort(np.array(ratios))
    assert np.all((ratios>0) & (ratios<100))
    omegas = np.concatenate([100-ratios[::-1], [100], 100+ratios])

    rname = "runoff[mm/yr]"
    pname = "rain[mm/yr]"
    ename = "evap[mm/yr]"
    pch = "rain_perc_change[%]"
    ech = "evap_perc_change[%]"

    sims = []

    # run scenarios
    for pw, ew in prod(omegas, repeat=2):
        model.inputs[:, 0] = inputs[:, 0]*pw/100
        model.inputs[:, 1] = inputs[:, 1]*ew/100
        model.initialise_fromdata()
        model.run()

        # Mean annual values
        q = model.outputs[iactive, 0].mean()*12
        p = model.inputs[iactive, 0].mean()*12
        e = model.inputs[iactive, 1].mean()*12

        dd = {\
            rname: q, \
            pname: p, \
            ename:e, \
            pch: pw, \
            ech: ew
        }
        sims.append(dd)

    # Store outputs
    sims = pd.DataFrame(sims)

    # Reset model
    model.inputs[:, 0] = inputs[:, 0]
    model.inputs[:, 1] = inputs[:, 1]
    model.initialise_fromdata()
    model.run()

    # Reference simulations
    i0 = (sims.loc[:, pch]==100) & (sims.loc[:, ech]==100)
    p0 = sims.loc[i0, pname].squeeze()
    e0 = sims.loc[i0, ename].squeeze()
    q0 = sims.loc[i0, rname].squeeze()

    # Compute elasticities
    elast = {"rain": {}, "evap":{}}
    for ratio in ratios:
        for vname in ["rain", "evap"]:
            if vname == "rain":
                cha, va = pch, pname
                chb, vb = ech, ename
                x0 = p0
            else:
                cha, va = ech, ename
                chb, vb = pch, pname
                x0 = e0

            i1 = (sims.loc[:, cha]==100-ratio) & (sims.loc[:, chb]==100)
            x1 = sims.loc[i1, va].squeeze()
            q1 = sims.loc[i1, rname].squeeze()

            i2 = (sims.loc[:, cha]==100+ratio) & (sims.loc[:, chb]==100)
            x2 = sims.loc[i2, va].squeeze()
            q2 = sims.loc[i2, rname].squeeze()

            el = (q2-q1)/(x2-x1)
            el_low = (q0-q1)/(x0-x1)
            el_high = (q2-q0)/(x2-x0)

            if relative:
                el *= x0/q0
                el_low *= x0/q0
                el_high *= x0/q0

            elast[vname][f"BOTH-{ratio:0.0f}%"] = el
            elast[vname][f"LOW-{ratio:0.0f}%"] = el_low
            elast[vname][f"HIGH-{ratio:0.0f}%"] = el_high

    for vname in ["rain", "evap"]:
        elast[vname] = pd.Series(elast[vname])

    return elast, sims


# Useful functions
def smooth(x, y, frac=1./3):
    if not HAS_STATSMODELS:
        raise ValueError("Cannot import statsmodels")
    x0, x1 = np.nanmin(x), np.nanmax(x)
    xx = np.linspace(x0, x1, 500)
    yy = sm.nonparametric.lowess(y, x, frac=frac, xvals=xx)
    return xx, yy


