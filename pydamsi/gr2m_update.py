import re
import math
import numpy as np
import pandas as pd

from itertools import product as prod

from scipy.stats import invgamma

from hydrodiy.stat import sutils
from hydrodiy.data.containers import Vector

from pygme.model import Model, ParamsVector
from pygme.factory import model_factory

from pydamsi import damsi_utils
import c_pydamsi

def get_interp_params_names():
    nm3 = [f"{i:02d}" for i in range(10)]
    nm2 = [f"{i:02d}" for i in range(6)]

    names = {
        "S": [f"S-{n}" for n in nm3],\
        "P3": [f"P3-{n}" for n in nm3], \
        "R": [f"R-{n}" for n in nm2], \
        "Q": [f"Q-{n}" for n in nm2]
    }
    return names


def get_interp_params_ini(X1, X2, Xr, \
                lamP, lamQ, lamE, nu):
    # Names
    names = get_interp_params_names()

    # All parameter are zero
    Sparams = pd.Series(np.zeros(10), index=names["S"])
    P3params = pd.Series(np.zeros(10), index=names["P3"])
    Rparams = pd.Series(np.zeros(6), index=names["R"])
    Qparams = pd.Series(np.zeros(6), index=names["Q"])

    params = {\
        "params": {}, \
        "GR2M": [X1, X2, Xr], \
        "info": {
            "lamP": lamP, \
            "lamQ": lamQ, \
            "lamE": lamE, \
            "nu": nu
        }
    }
    params["params"]["S"] = Sparams
    params["params"]["P3"] = P3params
    params["params"]["R"] = Rparams
    params["params"]["Q"] = Qparams
    return params


class GR2MUPDATE(Model):
    def __init__(self):
        # Config vector
        lnames = get_interp_params_names()
        lnames = lnames["S"]+lnames["P3"]+lnames["R"]+lnames["Q"]

        # First two config items are
        # Xr = routing store capacity
        # lamP = BoxCox exponent for rainfall perturbation
        # lamQ = BoxCox exponent for runoff perturbation
        # nu = BoxCox shift param
        # usebaseline = apply modification to production store or not
        # nodesS 1, 2, 3 = S approx nodes
        # nodesR 1, 2, 3 = R approx nodes
        nSnames = [f"nodeS{i}" for i in range(1, 4)]
        nRnames = [f"nodeR{i}" for i in range(1, 4)]
        cnames = ["notused", "lamP", "lamE", "lamQ", "nu"]\
                        + nSnames + nRnames + lnames
        # Default radius is set to 0 to match with default useradial=0
        cdef = [60, 0.0, 1.0, 0.2, 1.0] + [0.]*len(lnames)
        cmins = [1, -1., -1., 0., 0.] + [-1e100]*len(lnames)
        cmaxs = [500, 2., 2., 2., 10.] + [1e100]*len(lnames)
        config = Vector(cnames, cdef, cmins, cmaxs)

        # set reference value from GR2M evaluation
        pmodel = model_factory("GR2M")
        names = ["X1", "X2", "Xr"]
        defaults = pmodel.params.defaults.tolist()+[60.]
        mins = pmodel.params.mins.tolist()+[1.]
        maxs = pmodel.params.maxs.tolist()+[1000.]
        vect = Vector(names, defaults=defaults, \
                            mins=mins, maxs=maxs)
        params = ParamsVector(vect)

        # State vector
        snames = pmodel.states.names.tolist()
        states = Vector(snames, check_bounds=False)

        # Model
        super(GR2MUPDATE, self).__init__("GR2MUPDATE", \
            config, params, states, \
            ninputs=8, \
            noutputsmax=20)

        self.inputs_names = ["Rain", "PET", \
                                "tPdelta", "tEdelta", \
                                "Sdelta", "tP3delta", \
                                "Rdelta", "tQdelta"]
        self.outputs_names = ["Q", "S", "R", \
                                "F", "P3", "AE", \
                                "P", "E", \
                                "S_nocheck", "P3_nocheck", \
                                "R_nocheck", "Q_nocheck", \
                                "yS_baseline", "yP3_baseline", \
                                "yR_baseline", "yQ_baseline", \
                                "yS_update", "yP3_update", \
                                "yR_update", "yQ_update"]

        self.parent_model = pmodel


    @property
    def transP(self):
        return damsi_utils.Transform(lam=self.lamP, nu=self.nu)


    @property
    def transQ(self):
        return damsi_utils.Transform(lam=self.lamQ, nu=self.nu)


    @property
    def transE(self):
        return damsi_utils.Transform(lam=self.lamE, nu=self.nu)


    def initialise_fromdata(self):
        S0 = 0.5*self.X1
        R0 = self.Xr/3
        self.initialise(states=[S0, R0])


    def allocate(self, inputs, noutputs=8):
        # Adds zero perturbation to inputs if not defined
        nval, nvar = inputs.shape
        if nvar == 2:
            inputs = np.column_stack([inputs, np.zeros((nval, 6))])

        super(GR2MUPDATE, self).allocate(inputs, noutputs)


    def run(self):
        ierr = c_pydamsi.gr2m_update_run(self.istart, self.iend,
            self.config.values, \
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(f"Model {self.name},"+\
                    f" c_pydamsi.gr2m_update_run returns {ierr}")


    def get_parent_simulation(self, sini=None, index=None):
        """ Computes GR2M simulation using the same parameters and
            same inputs
        """
        model = self.parent_model
        model.allocate(self.inputs[:, :2], model.noutputsmax)
        model.X1 = self.X1
        model.X2 = self.X2
        model.Rcapacity = self.Xr
        if sini is None:
            model.initialise_fromdata()
        else:
            model.initialise(sini)
        model.run()
        return model.to_dataframe(include_inputs=True, index=index)


    def get_interp_params_indexes(self):
        """ Indexes within config vector """
        i0 = c_pydamsi.get_modif_params_start()
        assert self.config.names[i0] == "S-00"
        kndS = np.arange(i0-6, i0-3) # Nodes S
        kndR = np.arange(i0-3, i0) # Nodes R
        nS = 10
        kS = np.arange(nS) # Config S
        nR = 6
        kR = np.arange(nR) # Config R
        return i0, kndS, nS, kS, kndR, nR, kR


    def set_interp_params(self, interp_params=None):
        usebaseline = self.get_usebaseline()
        useradial = self.get_useradial()
        uselinpred = self.get_uselinpred()
        useconstraint = self.get_useconstraint()

        if interp_params is None:
            interp_params = get_interp_params_ini(\
                                    X1=self.X1, X2=self.X2, Xr=self.Xr, \
                                    lamP=self.lamP, lamE=self.lamE, \
                                    lamQ=self.lamQ, nu=self.nu)
        # Check keys
        assert isinstance(interp_params, dict)
        assert set(["info", "GR2M", "params"]) == set(interp_params.keys())
        keys = {"lamP", "lamQ", "lamE", "nu"}
        assert (set(interp_params["info"].keys()) & keys) == keys

        # Set GR2M params
        X1, X2, Xr = interp_params["GR2M"]
        self.X1 = X1
        self.X2 = X2
        self.Xr = Xr

        # Set config
        info = interp_params["info"]
        self.lamP = info["lamP"]
        self.lamQ = info["lamQ"]
        self.nu = info["nu"]

        # Set interpolation params
        params = interp_params["params"]
        i0, kndS, nS, kS, kndR, nR, kR = \
                        self.get_interp_params_indexes()
        if "S" in params:
            self.config.values[i0+kS] = np.array(params["S"])

        if "P3" in params:
            self.config.values[i0+nS+kS] = np.array(params["P3"])

        if "R" in params:
            self.config.values[i0+2*nS+kR] = np.array(params["R"])

        if "Q" in params:
            self.config.values[i0+2*nS+nR+kR] = np.array(params["Q"])


    def get_interp_params(self):
        i0, kndS, nS, kS, kndR, nR, kR = \
                self.get_interp_params_indexes()
        return {
            "params": {
                "S": self.config.values[i0+kS], \
                "P3": self.config.values[i0+nS+kS], \
                "R": self.config.values[i0+2*nS+kR], \
                "Q": self.config.values[i0+2*nS+nR+kR], \
            }, \
            "GR2M": [self.X1, self.X2, self.Xr, self.alphaP, self.alphaE], \
            "info": {
                "lamQ": self.lamQ, \
                "lamE": self.lamE, \
                "lamP": self.lamP, \
                "nu": self.nu
            }
        }


    def convert_interp_params_vect2dict(self, vect):
        cfg = self.config.values.copy()
        for cn in self.config.names:
            if re.search("^(S|P3|R|Q)", cn):
                v = vect.filter(regex=f"{cn}$")
                assert len(v) == 1
                self[cn] = v.iloc[0]

        betas = self.get_interp_params()
        # .. revert to original config
        self.config.values = cfg

        return betas



def get_interpolation_variables(sims, X1, X2, Xr):
    """ Computes interpolation variables:
        - Production store
            XS = [u=S/X1, phi=tanh(P/X1), psi=tanh(E/X1)]
            WS = [1, XS1, XS2, XS3, XS1**2, XS2**2, XS3**2,
                        XS1*XS2, XS1*XS2, XS2*XS3]

            S regression var : u = S/X1
            P3 regression var: p3n = (Rain-P3)/X1

        - Routing store
            XR = [v=R/Xr, alpha=P3/(P3+Xr)]
            WR = [1, XR1, XR2, XR1**2, XR2**2, XR1*XR2]

            y-R regression var : rn = 2R/(Xr+Q)
            y-Q regression var : qn = Q/(Xr+Q)
    """
    nval = len(sims)

    # Production var
    u = damsi_utils.gr2m_prod_S_raw2norm(X1, sims.S.shift(1))
    phi = damsi_utils.gr2m_prod_P_raw2norm(X1, sims.Rain)
    psi = damsi_utils.gr2m_prod_E_raw2norm(X1, sims.PET)

    # Production polynomial
    XS = np.column_stack([u, phi, psi])
    WS = 1e-8*np.random.uniform(-1, 1, size=(nval, 27))
    val1, val2, val3 = u.copy(), phi.copy(), psi.copy()
    WS[:, 0] = 1.
    WS[:, 1] = val1
    WS[:, 2] = val2
    WS[:, 3] = val3
    WS[:, 4] = val1**2
    WS[:, 5] = val2**2
    WS[:, 6] = val3**2
    WS[:, 7] = val1*val2
    WS[:, 8] = val1*val3
    WS[:, 9] = val2*val3

    # Routing var
    v = damsi_utils.gr2m_rout_Rstart_raw2norm(X2, Xr, sims.R.shift(1))
    alpha = damsi_utils.gr2m_rout_P3_raw2norm(X2, Xr, sims.P3)

    # Routing polynomial
    XR = np.column_stack([v, alpha])
    WR = 1e-8*np.random.uniform(-1, 1, size=(nval, 9))
    val1, val2 = v.copy(), alpha.copy()
    WR[:, 0] = 1
    WR[:, 1] = val1
    WR[:, 2] = val2
    WR[:, 3] = val1**2
    WR[:, 4] = val2**2
    WR[:, 5] = val1*val2

    # Response variables
    Yuend = damsi_utils.gr2m_prod_S_raw2norm(X1, sims.S)
    Yp3n = damsi_utils.gr2m_prod_P3_raw2norm(X1, sims.P3)
    Yvend = damsi_utils.gr2m_rout_Rend_raw2norm(X2, Xr, sims.R)
    Yqn = damsi_utils.gr2m_rout_Q_raw2norm(X2, Xr, sims.Q)

    interp_data = {\
        "data": {
            "index": sims.index, \
            "XS": XS, \
            "WS": WS, \
            "XR": XR, \
            "WR": WR, \
            "Yuend": Yuend, \
            "Yuend_baseline": damsi_utils.gr2m_S_fun_normalised(X1, *XS.T), \
            "Yp3n": Yp3n, \
            "Yp3n_baseline": damsi_utils.gr2m_P3_fun_normalised(X1, *XS.T), \
            "Yvend": Yvend, \
            "Yvend_baseline": damsi_utils.gr2m_R_fun_normalised(X2, Xr, *XR.T), \
            "Yqn": Yqn, \
            "Yqn_baseline": damsi_utils.gr2m_Q_fun_normalised(X2, Xr, *XR.T), \
        }, \
        "info": {
            "state_names": ["S", "P3", "R", "Q"],\
            "X1": X1, \
            "X2": X2, \
            "Xr": Xr
        }
    }
    return interp_data



def fit_interpolation(interp_data, \
            ical, \
            lamP, lamQ, lamE, nu, \
            nsamples=5000):
    # Check inputs

    data = interp_data["data"]
    info = interp_data["info"]
    index = data["index"]

    # initialiase
    states_names = info["state_names"]
    pnames = get_interp_params_names()
    X1 = info["X1"]
    X2 = info["X2"]
    Xr = info["Xr"]

    # .. initial parameters
    betas_ini = get_interp_params_ini(X1, X2, Xr, \
                        lamP=lamP, lamE=lamE, lamQ=lamQ,\
                        nu=nu)
    betas_ref = {\
        "params": {}, \
        "GR2M": [X1, X2, Xr], \
        "info": {
            "lamP": lamP, \
            "lamE": lamE, \
            "lamQ": lamQ, \
            "nu": nu
        }
    }

    for state in states_names:
        # .. get data
        if state in ["S"]:
            Xo = data["XS"][ical]
            Ybaseline = data["Yuend_baseline"][ical]
            W = data["WS"][ical]
            Ysim = data["Yuend"][ical]
            npreds = 10

        elif state == "P3":
            Xo = data["XS"][ical]
            Ybaseline = data["Yp3n_baseline"][ical]
            W = data["WS"][ical]
            Ysim = data["Yp3n"][ical]
            npreds = 10

        elif state == "R":
            Xo = data["XR"][ical]
            Ybaseline = data["Yvend_baseline"][ical]
            W = data["WR"][ical]
            Ysim = data["Yvend"][ical]
            npreds = 6

        elif state == "Q":
            Xo = data["XR"][ical]
            Ybaseline = data["Yqn_baseline"][ical]
            W = data["WR"][ical]
            Ysim = data["Yqn"][ical]
            npreds = 6

        # .. get regression data
        iok = np.all(~np.isnan(W), axis=1) & ~np.isnan(Ysim)\
                    & ~np.isnan(Ybaseline)
        X = W[iok][:, :npreds]
        y = np.array(Ysim[iok] - Ybaseline[iok])

        index_y = index[ical][iok]

        # .. initialise regression
        breg = damsi_utils.BayesianRegression(X, y)
        breg.solve()

        # Use posterior distribution mean
        pns = pnames[state]
        bm = pd.Series(0., index=pns)
        bb = breg.betan
        bm.iloc[:len(bb)] = bb

        # .. set full parameter as bayesian reg mean
        betas_ref["params"][state] = bm

        # .. store data
        dd = {\
            "breg": breg, \
            "index": index_y,
        }
        betas_ref["info"][f"{state}_diag"] = dd

    return betas_ref


