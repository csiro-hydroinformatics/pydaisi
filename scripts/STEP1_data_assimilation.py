#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2023-10-24 08:49:29.817157
## Comment : DAISI STEP 1
##           Ensemble Smoother data assimilation applied to the GR2M model
##
## ------------------------------

import sys, os, re, json, math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pygme import factory
from pydaisi import daisi_data, gr2m_update, gr2m_ensmooth,\
                        daisi_perf

from tqdm import tqdm

import importlib
importlib.reload(gr2m_ensmooth)

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="DAISI STEP 1 - data assimilation", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--debug", help="Debug mode", \
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

# Model calibrated in this script
# See pygme.models for a list of potential models
model_name = "GR2M"

# Objective functions
objfun_names = ["kge", "bc02"]

# Configure data assimilation
nens = 500

# .. assimilation variable box-cox transformation
lamP = 0.
lamE = 1.0
lamQ = 0.2
nu = 1.

# .. Config perturbation
alphae = 0.1

# .. create ensmooth config dictionary
# .. (reduction factor applied to stdev)
state_names = ["P", "E", "S", "P3", "R", "Q", "Q_obs"]
stdfacts = {n:alphae for n in state_names}

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "STEP1_data_assimilation"
fout.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / f"{basename}.log"
flog.parent.mkdir(exist_ok=True)
LOGGER = iutils.get_logger(basename, flog=flog, contextual=True, console=False)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Select siteids. All sites by default.
sites = daisi_data.get_sites()

if debug:
    #siteids_debug = [405218, 234201, 405240, 401013, 410038, 219017]
    siteids_debug = [405218]
    sites = sites.loc[siteids_debug]

# Calibration periods
periods = daisi_data.Periods()

# Calibration results
fparams = fout.parent / "STEP0_gr2m_calibration" / "calib_results.csv"
if not fparams.exists():
    errmess = "Calibration results do not exist. Run STEP0 script."
    raise ValueError(errmess)
params, _ = csv.read_csv(fparams)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
model = factory.model_factory(model_name)
nsites = len(sites)
perfs = []

for isite, (siteid, sinfo) in tqdm(enumerate(sites.iterrows()), \
                total=nsites, desc="Smoothing", disable=debug):
    LOGGER.context = f"{siteid} ({isite+1}/{nsites})"

    LOGGER.info("Load data")
    mthly = daisi_data.get_data(siteid)

    for objfun_name in objfun_names:
        # Assimilation output folder
        fassim = fout / f"assim_{objfun_name}"
        fassim.mkdir(exist_ok=True)

        fimg = fassim / "images"
        fimg.mkdir(exist_ok=True)

        # Run calibration for each period
        for calperiod in periods.periods.keys():

            if calperiod == "per3":
                # Skip calibration on the whole period.
                # Just cal/val
                continue

            LOGGER.info(f"Run {objfun_name} - calperiod {calperiod}")

            # Instanciate model
            model = gr2m_update.GR2MUPDATE()

            # Set transform
            model.lamQ = lamQ
            model.lamP = lamP
            model.lamE = lamE
            model.nu = nu

            # Calibration period
            calp = periods.get_periodset(calperiod)
            idxcal = calp.total.select_index(mthly.index)
            idxcal_active = calp.active.select_index(mthly.index)[idxcal]

            # Get streamflow data during calib
            Qobs = mthly.Qobs.loc[idxcal]
            Qobs[~idxcal_active] = np.nan
            Qobscal = Qobs.copy()

            obs = pd.DataFrame({"Q": Qobs})
            obscal = pd.DataFrame({"Q": Qobscal})

            # Initialise GR2M model using calibrated parameters
            model.allocate(mthly.loc[idxcal, ["Rain", "Evap"]])

            idxr = (params.INFO_siteid==siteid) \
                        & (params.INFO_objfun==objfun_name)\
                        & (params.INFO_calperiod==calperiod)\
                        & (params.INFO_model=="GR2M")
            if idxr.sum() != 1:
                errmess = "Cannot find params for " +\
                                f"{siteid}/{calperiod}/{objfun_name}"
                raise ValueError(errmess)

            X1 = params.loc[idxr, "PARAM_GR2M_X1"].squeeze()
            X2 = params.loc[idxr, "PARAM_GR2M_X2"].squeeze()
            Xr = 60.

            model.X1 = X1
            model.X2 = X2
            model.Xr = Xr

            model.initialise_fromdata()

            LOGGER.info("Run ENKS")
            # .. create ensmooth object
            ensmooth = gr2m_ensmooth.EnSmooth(model, \
                    obscal, stdfacts, debug, nens)

            # .. configure plotting
            ensmooth.plot_dir = fimg
            ensmooth.plot_ax_size = (20, 3)
            ensmooth.plot_freq = 1000
            y1 = calp.active.end.year-5
            y2 = calp.active.end.year
            ensmooth.plot_period = [y1, y2]

            # .. initialise object
            ensmooth.initialise()

            context = f"{siteid}_{objfun_name}_{calperiod}"
            message = f"EnSmooth {siteid} ({isite+1}/{nsites}) "+\
                        f"{objfun_name}-{calperiod}"
            ensmooth.run(context, message)

            # Retrive key data from ensmooth
            nstates, nens, Xa = ensmooth.nstates_assim, ensmooth.nens, ensmooth.Xa
            sims0 = ensmooth.sims0
            transQ = ensmooth.transQ
            transP = ensmooth.transP
            transE = ensmooth.transE
            cols = [f"Ens{iens:03d}" for iens in range(nens)]
            iQ = ensmooth.assim_states.index("Q")

            meta = {
                "INFO_siteid": int(siteid), \
                "INFO_calperiod": calperiod, \
                "INFO_objfun": objfun_name, \
                "INFO_nens": nens, \
                "CONFIG_lamQ": transQ.lam,\
                "CONFIG_nuQ": transQ.nu,\
                "CONFIG_lamE": transE.lam,\
                "CONFIG_nuE": transE.nu,\
                "CONFIG_lamP": transP.lam,\
                "CONFIG_nuP": transP.nu,\
                "CONFIG_alphae": alphae, \
                "PARAM_GR2M_X1": X1, \
                "PARAM_GR2M_X2": X2, \
                "PARAM_GR2M_Xr": Xr
            }

            LOGGER.info("Store - Xa")
            snames = ensmooth.assim_states
            nparams = 0 # no parameter assim

            # .. back transform corrected state data to
            #   facilitate data interpretation
            if "P" in snames:
                iP = snames.index("P")
                Xa[nparams+iP::nstates] = \
                                transP.backward(Xa[nparams+iP::nstates]) #P

            if "P3" in snames:
                iP3 = snames.index("P3")
                Xa[nparams+iP3::nstates] = \
                                transP.backward(Xa[nparams+iP3::nstates]) #P3

            if "E" in snames:
                iE = snames.index("E")
                Xa[nparams+iE::nstates] = \
                                transP.backward(Xa[nparams+iE::nstates]) #E

            if "AE" in snames:
                iAE = snames.index("AE")
                Xa[nparams+iAE::nstates] = \
                                transE.backward(Xa[nparams+iAE::nstates]) #AE

            HXa = Xa[nparams+iQ::nstates]
            Xa[nparams+iQ::nstates] = transQ.backward(HXa) #Q
            Xa = pd.DataFrame(Xa, columns=cols)
            time = np.repeat(Qobs.index, nstates)
            Xa.loc[:, "time"] = time

            sn = np.repeat(np.array(snames)[None, :], len(HXa), axis=0).ravel()
            Xa.loc[:, "state"] = sn

            fn = f"ensmooth_Xa_{siteid}_{calperiod}.csv"
            fxa = fassim / fn
            comment = {"comment": "Enks Xa data"}
            comment.update(meta)
            csv.write_csv(Xa, fxa, comment, \
                    source_file, write_index=True)

            # Process openloop
            LOGGER.info("Store - Xf")
            tend = (Xa.shape[0]-nparams)//nstates
            Xf = ensmooth.openloop(0, tend)

            if "P" in snames:
                Xf[iP::nstates] = transP.backward(Xf[iP::nstates]) #P

            if "P3" in snames:
                Xf[iP3::nstates] = transP.backward(Xf[iP3::nstates]) #P3

            if "E" in snames:
                Xf[iE::nstates] = transE.backward(Xf[iE::nstates]) #E

            if "AE" in snames:
                Xf[iAE::nstates] = transE.backward(Xf[iAE::nstates]) #AE

            HXf = Xf[iQ::nstates]
            Xf[iQ::nstates] = transQ.backward(HXf) #Q
            Xf = pd.DataFrame(Xf, columns=cols)
            Xf.loc[:, "time"] = time[nparams:]
            Xf.loc[:, "state"] = sn[nparams:]

            fn = f"ensmooth_Xf_{siteid}_{calperiod}.csv"
            fxf = fassim / fn
            comment = {"comment": "Enks Xf data"}
            comment.update(meta)
            csv.write_csv(Xf, fxf, comment, \
                    source_file, write_index=True)

            # HXa
            LOGGER.info("Store - HXa")
            HXa = pd.DataFrame(HXa, index=Qobs.index, columns=cols)
            HXa.loc[:, "Qobs"] = Qobs
            HXa.loc[:, "Qsim"] = sims0.Q.values
            HXa.loc[:, "P3sim"] = sims0.P3.values
            HXa.loc[:, "Ssim"] = sims0.S.values
            HXa.loc[:, "Rsim"] = sims0.R.values
            HXa.loc[:, "Rain"] = model.inputs[:, 0]
            HXa.loc[:, "Evap"] = model.inputs[:, 1]

            HXa.loc[:, "ISCAL"] = 0
            HXa.loc[idxcal_active, "ISCAL"] = 1

            ens = HXa.filter(regex="Ens", axis=1)
            obs = HXa.Qobs
            _, nrmse, ksp, pits = daisi_perf.ensemble_metrics(obs, ens)
            log10ksp = math.log10(max(1e-10, ksp))

            perfs.append({"siteid": siteid, "calperiod": calperiod, \
                            "objfun": objfun_name, \
                            "nrmse": nrmse, "ksp": ksp})

            LOGGER.info(f"DAPERF: NR={nrmse:0.2f} KS={ksp:2.2e}")

            fhxa = fxa.parent / f"{re.sub('Xa', 'HXa', fxa.stem)}.csv"
            comment = {\
                "comment": "Enks HXa data", \
                "METRIC_CAL_NRMSERATIO-DA": nrmse, \
                "METRIC_CAL_KSLOG10PV-DA":log10ksp, \
            }
            comment.update(meta)
            csv.write_csv(HXa, fhxa, comment, \
                    source_file, write_index=True)


# Store results
perfs = pd.DataFrame(perfs)
fr = fout / f"assim_results.csv"
csv.write_csv(perfs, fr, "Data Assimilation results", \
                source_file, compress=False, line_terminator="\n")


LOGGER.info("Process completed")

