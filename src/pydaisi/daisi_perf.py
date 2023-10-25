import math, re
from itertools import product as prod
import numpy as np
import pandas as pd
from scipy.stats import norm, t as studentt

from hydrodiy.io import csv, iutils
from hydrodiy.data import dutils
from hydrodiy.stat import metrics, transform

from pydaisi import daisi_utils


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

    elif n == "NRMSERATIO":
        nm = r"$N_R$"
        lnm = f"Normalised RMSE ratio ({nm})"

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



def deterministic_metrics(Qobs, Qsim, time_full, ieval, calval, name, perfs=None):
    # Get data
    time = time_full[ieval]
    qo = np.array(Qobs)[ieval]
    qs = np.maximum(np.array(Qsim)[ieval], 0)
    perfs = {} if perfs is None else perfs

    # Bias
    bias = metrics.bias(qo, qs)
    perfs[f"METRIC_{calval}_MEANANNUALOBS"] = np.mean(qo)*12
    perfs[f"METRIC_{calval}_MEANANNUALSIM_{name}"] = np.mean(qs)*12
    perfs[f"METRIC_{calval}_BIAS_{name}"] = bias
    perfs[f"METRIC_{calval}_ABSBIAS_{name}"] = 1.-abs(bias)

    # NSE
    perfs[f"METRIC_{calval}_NSE_{name}"] = metrics.nse(qo, qs)

    # KGE
    perfs[f"METRIC_{calval}_KGE_{name}"] = metrics.kge(qo, qs)

    # Split KGE
    # see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2017WR022466
    years = np.unique(time.year)
    skge = [metrics.kge(qo[time.year==y], qs[time.year==y]) for y in years]
    skge = np.array(skge).mean()
    perfs[f"METRIC_{calval}_SPLITKGE_{name}"] = skge


    # Multi year PMR (see Royer-Gaspard, 2021, page 98)
    o, s = pd.Series(Qobs, index=time_full), pd.Series(Qsim, index=time_full)
    myobs = o[ieval].mean()
    ny = 5
    ynobs = o.rolling(ny).mean()[ieval]
    ynsim = s.rolling(ny).mean()[ieval]
    ynerr = ynsim-ynobs
    mynerr = ynerr.mean()
    perfs[f"METRIC_{calval}_PMR{ny}Y_{name}"] = \
                            float(2*np.abs(ynerr-mynerr).mean()/myobs)

    # Offset to avoid zero flow issue
    eps = 1.0

    # Transformed nse
    tr = lambda x: np.log(eps+x)
    pname = f"METRIC_{calval}_NSELOG_{name}"
    perfs[pname] = metrics.nse(tr(qo), tr(qs))

    tr = lambda x: np.sqrt(eps+x)
    pname = f"METRIC_{calval}_NSESQRT_{name}"
    perfs[pname] = metrics.nse(tr(qo), tr(qs))

    tr = lambda x: 1-eps/(eps+x)
    pname = f"METRIC_{calval}_NSERECIP_{name}"
    perfs[pname] = metrics.nse(tr(qo), tr(qs))

    # Quantile errors
    cov = 100
    fdc1, _ = metrics.relative_percentile_error(qo, qs,\
                                              [0, cov], eps=eps)
    pname = f"METRIC_{calval}_FDCFIT{cov}_{name}"
    perfs[pname] = fdc1

    pname = f"METRIC_{calval}_ABSFDCFIT{cov}_{name}"
    perfs[pname] = 1-abs(fdc1)

    return perfs


def elasticity_metrics(model, ieval, calval, name, perfs=None):
    perfs = {} if perfs is None else perfs

    for relative in [True, False]:
        el, sims = daisi_utils.model_elasticity(model, ieval, relative)

        etype = "rel" if relative else "abs"
        perfs[f"METRIC_{calval}_ELAST{etype}RAIN_{name}"] = el["rain"]["BOTH-10%"]
        perfs[f"METRIC_{calval}_ELAST{etype}EVAP_{name}"] = el["evap"]["BOTH-10%"]

    return perfs


def normalised_rmse_ratio(obs, ens):
    """ Normalised root-mean-squared error ratio.
    See Thiboult, A. and F. Anctil (2015). "On the difficulty to optimally
    implement the Ensemble Kalman filter: An experiment based on many
    hydrological models and catchments." JOURNAL OF HYDROLOGY 529: 1147-1160.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        ensemble forecast data, [n,p] array

    Returns
    -----------
    ratio : float
        Ratio of spread vs rmse
    rmse_of_mean_ens: float
        RMSE of mean ensemble
    mean_of_spread: float
        Mean of ensemble error standard deviation
    """
    # Check inputs
    obs, ens = np.array(obs), np.array(ens)
    assert obs.ndim == 1
    assert ens.ndim == 2
    assert obs.shape[0] == ens.shape[0]
    R = ens.shape[1]

    # Compute numerator and denominator of ratio
    squared_error_of_mean_ens = (ens.mean(axis=1)-obs)**2
    rmse_of_mean_ens = math.sqrt(squared_error_of_mean_ens.mean())

    err = ens-obs[:, None]
    c = math.sqrt((R+1)/(2*R))
    mean_of_spread = (np.sqrt((err*err).mean(axis=0))).mean()*c

    ratio = rmse_of_mean_ens/mean_of_spread
    return ratio, rmse_of_mean_ens, mean_of_spread


def ensemble_metrics(obs, ens):
    # Check inputs
    obs, ens = np.array(obs), np.array(ens)
    assert obs.ndim == 1
    assert ens.ndim == 2
    assert obs.shape[0] == ens.shape[0]

    iok = ~np.isnan(obs)
    if iok.sum()<5:
        return np.nan, np.nan, np.nan

    obs = obs[iok]
    ens = ens[iok]

    # compute metrics
    c, _ = metrics.crps(obs, ens)
    crps_score = 1-c.crps/c.uncertainty

    nrmse_ratio, _, _ = normalised_rmse_ratio(obs, ens)

    pits, _ = metrics.pit(obs, ens)
    _, ks_pvalue, _ = metrics.alpha(obs, ens, type="KS")

    return crps_score, nrmse_ratio, ks_pvalue, pits

