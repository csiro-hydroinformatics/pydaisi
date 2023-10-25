#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2023-10-24 08:49:29.817157
## Comment : DAISI STEP 2
##           Model structure update by fitting update coefficients
##
## ------------------------------

import sys, os, re, json, math
import argparse
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pygme.models import gr2m
from pydaisi import daisi_data, daisi_perf, daisi_utils,\
                        gr2m_update

from tqdm import tqdm

from select_sites import select_sites

import importlib
importlib.reload(gr2m_update)

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="DAISI STEP 2 - update fit", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--debug", help="Debug mode", \
                    action="store_true", default=False)
parser.add_argument("-t", "--taskid", help="Task id", \
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of batch processes", \
                    type=int, default=4)
parser.add_argument("-fo", "--folder_output", help="Output folder", \
                    type=str, default=None)
args = parser.parse_args()

debug = args.debug
taskid = args.taskid
nbatch = args.nbatch

folder_output = args.folder_output
if not folder_output is None:
    folder_output = Path(folder_output)
    assert folder_output.exists()

# Model calibrated in this script
# See pygme.models for a list of potential models
model_name = "GR2M"

# Objective functions
objfun_names = ["kge", "bc02"]

calperiods = ["per1", "per2"]

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "STEP2_model_structure_update"
if not folder_output is None:
    fout = folder_output / "STEP2_model_structure_update"
fout.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / f"{basename}.log"
if not folder_output is None:
    flog = folder_output / "logs" / f"{basename}.log"
flog.parent.mkdir(exist_ok=True)
LOGGER = iutils.get_logger(basename, flog=flog, contextual=True, console=False)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Select siteids. All sites by default.
sites = select_sites(daisi_data.get_sites(), debug, nbatch, taskid)

# Calibration periods
periods = daisi_data.Periods()

fassim = fout.parent / "STEP1_data_assimilation"

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
nsites = len(sites)

for isite, (siteid, sinfo) in tqdm(enumerate(sites.iterrows()), \
                total=nsites, desc="Smoothing", disable=debug):
    LOGGER.context = f"{siteid} ({isite+1}/{nsites})"

    LOGGER.info("Load data")
    mthly = daisi_data.get_data(siteid)

    for objfun_name, calperiod in prod(objfun_names, calperiods):
        LOGGER.info("")
        LOGGER.info(f"{objfun_name} - Period {calperiod}")

        ffit = fout / f"updatefit_{objfun_name}"
        ffit.mkdir(exist_ok=True)

        # Assimilation output folder
        fn = f"ensmooth_Xa_{siteid}_{calperiod}.csv"
        fxa = fassim / "" / f"assim_{objfun_name}" / fn
        Xa, meta = csv.read_csv(fxa, index_col="", parse_dates=True)
        nens = int(meta["info_nens"])
        X1 = float(meta["param_gr2m_x1"])
        X2 = float(meta["param_gr2m_x2"])
        Xr = float(meta["param_gr2m_xr"])

        lamP = float(meta["config_lamp"])
        lamE = float(meta["config_lame"])
        lamQ = float(meta["config_lamq"])
        nu = float(meta["config_nup"])

        fhxa = fxa.parent / re.sub("Xa", "HXa", fxa.name)
        HXa, _ = csv.read_csv(fhxa, index_col="", parse_dates=True)

        # Get time series data
        ical = HXa.Qobs.notnull() & (HXa.ISCAL==1)
        Rain_cal = HXa.Rain
        Evap_cal = HXa.Evap
        inputs_cal = np.column_stack([Rain_cal, Evap_cal])

        mthly = daisi_data.get_data(siteid)
        Rain_val = mthly.Rain
        Evap_val = mthly.Evap
        inputs_val = np.column_stack([Rain_val, Evap_val])
        Qobs = mthly.Qobs

        # Rainfall-runoff models
        # .. GR2M
        gr = gr2m.GR2M()
        gr.params.values = [X1, X2]

        gr.allocate(inputs_cal, gr.noutputsmax)
        gr.initialise_fromdata()
        gr.run()
        gsims_cal = gr.to_dataframe(index=Rain_cal.index).copy()

        gr.allocate(inputs_val, gr.noutputsmax)
        gr.initialise_fromdata()
        gr.run()
        gsims_val = gr.to_dataframe(index=Rain_val.index).copy()

        # Names and number of DA states
        assim_states = [s for s in Xa.state.unique() \
                            if not s in ["P", "E", "AE"]]

        LOGGER.info("Extract ensembles")
        ens = {}
        for state in assim_states:
            if re.search("^(X|alpha)", state):
                continue
            e = Xa.loc[Xa.state==state].filter(regex="Ens|time", axis=1)
            e = e.set_index("time")
            if state in ["S", "R"]:
                e = e.shift(-1)

            ens[state] = e


        LOGGER.info("Fit update to ensmoother ensembles")
        betas = []
        for iens in range(nens):
            if iens%100 == 0:
                LOGGER.info(f".. dealing with ens {iens:3d}/{nens}")

            # get ensemble data
            cn = f"Ens{iens:03d}"
            esims = pd.DataFrame({st: ens[st].loc[:, cn] \
                                        for st in assim_states})
            if "P" in ens:
                esims.loc[:, "Rain"] = ens["P"].loc[:, cn]
            else:
                esims.loc[:, "Rain"] = Rain_cal

            # Complete potentially missing data
            esims.loc[:, "PET"] = Evap_cal
            for state in assim_states:
                if not state in esims:
                    esims.loc[:, state] = gsims_cal.loc[:, state]

            esims = esims.astype(np.float64)

            # get interpolation data
            interp_data = gr2m_update.get_interpolation_variables(esims, \
                                        X1, X2, Xr)

            # Fit interpolation parameters
            pens = gr2m_update.fit_interpolation(interp_data, \
                                lamP=lamP, lamE=lamE, lamQ=lamQ, \
                                nu=nu, ical=ical)

            # Store parameters
            betas.append(pens)

        # Compute params estimator (mean)
        mparams = {s: pens["params"][s]*0. for s in assim_states}
        for iens in range(nens):
            for s in assim_states:
                # .. modif params
                mparams[s] += betas[iens]["params"][s]

        for s in assim_states:
             mparams[s] /= nens

        betas_mean = {
            "info":{"lamP": lamP, "lamE": lamE, "lamQ": lamQ, "nu": nu}
        }
        betas_mean["params"] = mparams
        betas_mean["GR2M"] = [X1, X2, Xr]

        LOGGER.info("Run updated sims")
        model = gr2m_update.GR2MUPDATE()
        model.allocate(inputs_val, model.noutputsmax)
        model.set_interp_params(betas_mean)
        model.initialise_fromdata()
        model.run()

        # Export sim
        msims = model.to_dataframe(include_inputs=True, index=mthly.index)
        cc = [cn for cn in msims.columns if not re.search("delta", cn)]
        msims = msims.loc[:, cc]
        msims.columns = [cn if cn in ["Rain", "PET"] else f"GR2MUPDATE_{cn}" \
                                for cn in msims.columns]
        # .. obs
        msims.loc[:, "Qobs"] = Qobs

        # .. gr2m sims
        for cn in model.outputs_names:
            if re.search("^P$|^E$|nocheck$|update$|baseline$", cn):
                continue
            msims.loc[:, f"GR2M_{cn}"] = gsims_val.loc[:, cn]

        # .. add cal/val periods
        calp = periods.get_periodset(calperiod)
        idxcal = calp.active.select_index(msims.index)
        msims.loc[:, "ISCAL"] = 0
        msims.loc[idxcal, "ISCAL"] = 1

        valp = periods.get_validation(calperiod)
        idxval = valp.active.select_index(msims.index)
        msims.loc[:, "ISVAL"] = 0
        msims.loc[idxval, "ISVAL"] = 1

        comments = {"comment": "Ensmoother fit results"}
        comments.update(meta)

        fs = ffit / f"updatefit_sims_{siteid}_{calperiod}.csv"
        csv.write_csv(msims, fs, comments, \
                source_file, write_index=True)

        # Store params
        betas_mean["info"]["siteid"] = siteid
        betas_mean["info"]["calperiod"] = calperiod
        betas_mean["info"]["objfun"] = objfun_name

        fp = ffit / f"updatefit_params_{siteid}_{calperiod}.json"
        for k in betas_mean["params"].keys():
            betas_mean["params"][k] = betas_mean["params"][k].to_dict()

        with fp.open("w") as fo:
            json.dump(betas_mean, fo, indent=4)


LOGGER.info("Process completed")

