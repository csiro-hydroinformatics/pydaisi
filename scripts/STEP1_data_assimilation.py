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

from pygme import calibration, factory
from pydaisi import daisi_data

from tqdm import tqdm

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
ensmooth = 1 # Run ensemble smoother (EnKS = 0)
assim_states = 2 # Prod and routing (no inputs)
clip = 0


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
    siteids_debug = [405218, 234201, 405240, 401013, 410038, 219017]
    sites = sites.loc[siteids_debug]

# Calibration periods
periods = daisi_data.Periods()

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
model = factory.model_factory(model_name)
nsites = len(sites)
results = []

for i, (siteid, row) in tqdm(enumerate(sites.iterrows()), \
                total=nsites, desc="Calibrating"):
    LOGGER.context = f"{siteid} ({i+1}/{nsites})"

    LOGGER.info("Load data")
    mthly = daisi_data.get_data(siteid)

    for objfun_name in objfun_names:
        # Get objective function object
        oname = re.sub("bc02", "bc0.2_0.1", objfun_name)
        objfun = factory.objfun_factory(oname)

        # Calibration output folder
        fcalib = fout / f"calibration_{objfun_name}"
        fcalib.mkdir(exist_ok=True)

        # Run calibration for each period
        for pername in periods.periods.keys():

            if pername == "per3":
                # Skip calibration on the whole period.
                # Just cal/val
                continue

            LOGGER.info("")
            LOGGER.info(f"***** Period {pername} *****")

            # Get calibration object
            warmup = periods.warmup_years*12
            calib = factory.calibration_factory(model_name,\
                                        nparamslib=nparamslib, \
                                        warmup=warmup, \
                                        objfun=objfun)

            # Select calib data
            incal, ocal, itotal, iactive, ical \
                        = daisi_data.get_inputs_and_obs(mthly, pername)

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
            pval = periods.get_validation(pername)
            ival = pval.active.select_index(mthly.index[itotal])
            sims.loc[:, "ival_active"] = 0
            sims.loc[ival, "ival_active"] = 1

            # Store data
            meta = {
                "INFO_model": model_name, \
                "INFO_warmup": periods.warmup_years, \
                "INFO_siteid": int(siteid), \
                "INFO_objfun": objfun_name, \
                "INFO_calperiod": pername, \
                "INFO_calnval": int(pd.notnull(ocal).sum()), \
                "INFO_nparamslib": nparamslib
            }
            if hasattr(objfun, "trans"):
                meta["CONFIG_objfun_lam"] = objfun.trans.lam
                meta["CONFIG_objfun_nu"] = objfun.trans.nu

            for pname, value in zip(model.params.names, final):
                meta[f"PARAM_{model_name}_{pname}"] = float(value.round(4))


            fs = fcalib / f"sim_{objfun_name}_{siteid}_{pername}.csv"
            csv.write_csv(sims, fs, "Calibrated simulations", \
                            source_file, write_index=True, \
                            line_terminator="\n")

            results.append(meta)

# Store results
results = pd.DataFrame(results)
fr = fout / f"calib_results.csv"
csv.write_csv(results, fr, "Calibrated results", \
                source_file, compress=False, line_terminator="\n")


LOGGER.info("Process completed")

