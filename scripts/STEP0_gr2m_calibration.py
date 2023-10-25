#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2023-10-24 08:49:29.817157
## Comment : Calibration of the GR2M model
##
## ------------------------------

import sys, os, re, json, math
import argparse
from itertools import product as prod
from pathlib import Path

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pygme import calibration, factory
from pydaisi import daisi_data

from select_sites import select_sites

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="Calibration of the GR2M model", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--debug", help="Debug mode (restricted site list)", \
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

# Number of parameters tested as part of calibration algorithm
nparamslib = 10000

if debug:
    nparamslib = 500

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "STEP0_gr2m_calibration"
if not folder_output is None:
    fout = folder_output / "STEP0_gr2m_calibration"
fout.mkdir(exist_ok=True, parents=True)

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
model = factory.model_factory(model_name)
nsites = len(sites)

for isite, (siteid, sinfo) in enumerate(sites.iterrows()):
    LOGGER.context = f"{siteid} ({isite+1}/{nsites})"

    LOGGER.info("Load data")
    mthly = daisi_data.get_data(siteid)

    for objfun_name, calperiod in prod(objfun_names, calperiods):
        LOGGER.info("")
        LOGGER.info(f"{objfun_name} - Period {calperiod}")

        # Get objective function object
        oname = re.sub("bc02", "bc0.2_0.1", objfun_name)
        objfun = factory.objfun_factory(oname)

        # Calibration output folder
        fcalib = fout / f"calibration_{objfun_name}"
        fcalib.mkdir(exist_ok=True)

        # Get calibration object
        warmup = periods.warmup_years*12
        calib = factory.calibration_factory(model_name,\
                                    nparamslib=nparamslib, \
                                    warmup=warmup, \
                                    objfun=objfun)

        # Select calib data
        incal, ocal, itotal, iactive, ical \
                    = daisi_data.get_inputs_and_obs(mthly, calperiod)

        # Calibrate
        final, ofun, _, ofun_explore = calib.workflow(ocal, \
                                    incal, ical=ical)

        # Run simulation over the whole period
        inall, oall, itotal, iactiveall, ievalall \
                        = daisi_data.get_inputs_and_obs(mthly, "per3")

        model.allocate(inall, model.noutputsmax)
        model.params.values = final
        model.initialise()
        model.run()
        sims = model.to_dataframe(mthly.index[itotal], True)

        sims.loc[:, "Qobs"] = mthly.Qobs[itotal]
        sims.loc[:, "ical_active"] = 0
        sims.loc[iactive[itotal], "ical_active"] = 1

        # Identify validation period
        pval = periods.get_validation(calperiod)
        ival = pval.active.select_index(mthly.index[itotal])
        sims.loc[:, "ival_active"] = 0
        sims.loc[ival, "ival_active"] = 1

        # Store data
        meta = {
            "INFO_model": model_name, \
            "INFO_warmup": periods.warmup_years, \
            "INFO_siteid": int(siteid), \
            "INFO_objfun": objfun_name, \
            "INFO_calperiod": calperiod, \
            "INFO_calnval": int(pd.notnull(ocal).sum()), \
            "INFO_nparamslib": nparamslib
        }
        if hasattr(objfun, "trans"):
            meta["CONFIG_objfun_lam"] = objfun.trans.lam
            meta["CONFIG_objfun_nu"] = objfun.trans.nu

        for pname, value in zip(model.params.names, final):
            meta[f"PARAM_{model_name}_{pname}"] = float(value.round(4))


        fs = fcalib / f"sim_{objfun_name}_{siteid}_{calperiod}.csv"
        csv.write_csv(sims, fs, meta, \
                        source_file, write_index=True, \
                        line_terminator="\n")

        fp = fs.parent / f"{fs.stem}.json"
        with fp.open("w") as fo:
            json.dump(meta, fo, indent=4)


LOGGER.info("Process completed")

