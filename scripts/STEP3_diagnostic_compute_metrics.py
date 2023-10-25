#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2023-10-25 Wed 03:38 PM
## Comment : DAISI STEP 3
##           Compute diagnostic metrics
##
## ------------------------------

import sys, os, re, json, math
import argparse
from pathlib import Path
from itertools import product as prod

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pygme.models import gr2m
from pydaisi import daisi_data, daisi_perf, daisi_utils,\
                        gr2m_update

from select_sites import select_sites

import importlib
importlib.reload(gr2m_update)

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="DAISI STEP 3 - compute diagnostic metrics", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--debug", help="Debug mode (restricted site list)", \
                    action="store_true", default=False)
parser.add_argument("-t", "--taskid", help="Site batch number (task id)",\
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of site batches", \
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

fout = froot / "outputs" / "STEP3_diagnostic"
if not folder_output is None:
    fout = folder_output / "STEP3_diagnostic"
fout.mkdir(exist_ok=True, parents=True)

fmetrics = fout / "metrics"
fmetrics.mkdir(exist_ok=True)

ffit = fout.parent / "STEP2_model_structure_update"

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / f"{basename}_TASK{taskid}.log"
if not folder_output is None:
    flog = folder_output / "logs" / f"{basename}_TASK{taskid}.log"
flog.parent.mkdir(exist_ok=True)
LOGGER = iutils.get_logger(basename, flog=flog, contextual=True, console=False)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Select siteids. All sites by default.
sites = select_sites(daisi_data.get_sites(), debug, nbatch, taskid)

# Calibration periods
periods = daisi_data.Periods()

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
nsites = len(sites)
perfs = []

for isite, (siteid, sinfo) in enumerate(sites.iterrows()):
    LOGGER.context = f"{siteid} ({isite+1}/{nsites})"

    LOGGER.info("Load data")
    mthly = daisi_data.get_data(siteid)

    results = []
    for objfun_name, calperiod in prod(objfun_names, calperiods):
        LOGGER.info("")
        LOGGER.info(f"{objfun_name} - Period {calperiod}")

        ff = ffit / f"updatefit_{objfun_name}"

        # Load update results
        fs = ff / f"updatefit_sims_{siteid}_{calperiod}.csv"
        msims, meta = csv.read_csv(fs, index_col="", parse_dates=True)
        meta = {k.upper(): v for k, v in meta.items() \
                        if re.search("config|info|param", k)}

        fp = ff / f"updatefit_params_{siteid}_{calperiod}.json"
        with fp.open("r") as fo:
            update_coefficients = json.load(fo)

        X1, X2, Xr = update_coefficients["GR2M"]

        for state in update_coefficients["params"].keys():
            se = update_coefficients["params"][state]
            update_coefficients["params"][state] = pd.Series(se)

        model = gr2m_update.GR2MUPDATE()
        inputs = np.ascontiguousarray(msims.loc[:, ["Rain", "PET"]])
        model.allocate(inputs)
        model.set_interp_params(update_coefficients)

        # .. GR2M
        gr = gr2m.GR2M()
        gr.X1 = X1
        gr.X2 = X2
        gr.allocate(inputs)

        # Run perf in cal/val modes
        for calval in ["CAL", "VAL"]:
            LOGGER.info(f"compute {calval} perf")
            idxe = (msims.loc[:, f"IS{calval}"]==1) \
                    & msims.Qobs.notnull()
            idxe = idxe.values
            qo = msims.Qobs.values

            for mname in ["GR2M", "GR2MUPDATE"]:
                # get data
                qs = msims.loc[:, f"{mname}_Q"].values

                # Standard metrics
                daisi_perf.deterministic_metrics(qo, qs, \
                                    msims.index, idxe, \
                                    calval, mname, perfs=meta)

                # Elasticity metrics
                if mname == "GR2M":
                    m = gr
                elif mname == "GR2MUPDATE":
                    m = model
                else:
                    raise ValueError()

                model.initialise_fromdata()
                daisi_perf.elasticity_metrics(m, idxe, \
                                calval, mname, perfs=meta)

        results.append(meta)

    LOGGER.info("Store final results")
    dfr = pd.DataFrame(results)
    fr = fmetrics / f"metrics_{siteid}.csv"
    csv.write_csv(dfr, fr, "Enksfit perf", source_file)



LOGGER.info("Process completed")

