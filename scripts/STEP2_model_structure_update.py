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

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pydaisi import daisi_data, daisi_perf, daisi_utils,\
                        gr2m_update

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

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "STEP2_data_assimilation"
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

fassim = fout.parent / "STEP1_data_assimilation"

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
nsites = len(sites)
perfs = []

for isite, (siteid, sinfo) in tqdm(enumerate(sites.iterrows()), \
                total=nsites, desc="Smoothing", disable=debug):
    LOGGER.context = f"{siteid} ({isite+1}/{nsites})"

    LOGGER.info("Load data")
    mthly = daisi_data.get_data(siteid)

    for objfun_name in objfun_names:
        # Assimilation output folder
        fda = fassim / f"assim_{objfun_name}"

        # TODO !

        fhxa = fxa.parent / f"{re.sub('Xa', 'HXa', fxa.stem)}.zip"
        HXa, meta = csv.read_csv(fhxa, index_col="", parse_dates=True)

        meta = {k: v for k, v in meta.items() \
                    if re.search("^(config|info|param|metric)", k)}

        nens = int(meta["info_nens"])
        X1 = float(meta["param_gr2m_x1"])
        X2 = float(meta["param_gr2m_x2"])
        Xr = float(meta["param_gr2m_xr"])

        # Get time series data
        ical = HXa.Qobs.notnull() & (HXa.ISCAL==1)
        Rain_cal = HXa.Rain
        Evap_cal = HXa.Evap
        inputs_cal = np.column_stack([Rain_cal, Evap_cal])

        mthly = nonstat_data.get_data(siteid)
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

        # GR2M DA
        grda = gr2m_modif.GR2MMODIF()
        grda.allocate(inputs_val, grda.noutputsmax)
        grda.set_interp_params()

        # GR2M post-processed -> linear correction on sqrt transform
        y = np.sqrt(1+HXa.Qobs[ical])
        x = np.sqrt(1+gsims_cal.Q[ical])
        X = np.column_stack([np.ones_like(x), x])
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-6)
        xx = np.sqrt(1+gsims_val.Q)
        XX = np.column_stack([np.ones_like(xx), xx])
        yy = XX.dot(theta)
        gpsims_val = pd.DataFrame({"Q": np.maximum(0, yy**2-1)}, \
                                            index=gsims_val.index)

        # .. wapaba
        wap = wapaba.WAPABA()
        idxr = (rrperfs.INFO_siteid==siteid) \
                    & (rrperfs.INFO_objfun==modobjfun)\
                    & (rrperfs.INFO_calperiod==calperiod)\
                    & (rrperfs.INFO_model=="WAPABA")
        assert idxr.sum() == 1
        wap.allocate(inputs_val, wap.noutputsmax)
        for pname in wap.params.names:
            setattr(wap, pname, \
                rrperfs.loc[idxr, f"PARAM_WAPABA_{pname}"].squeeze())

        wap.initialise_fromdata()
        wap.run()
        wsims_val = wap.to_dataframe(index=Rain_val.index).copy()

        # Names and number of DA states
        assim_states = [s for s in Xa.state.unique() \
                            if not s in ["P", "E", "AE", \
                                    "X1", "X2", "Xr", \
                                    "alphaP", "alphaE"]]

        # Extract ensembles
        ens = {}
        for state in assim_states:
            if re.search("^(X|alpha)", state):
                continue
            e = Xa.loc[Xa.state==state].filter(regex="Ens|time", axis=1)
            e = e.set_index("time")
            if state in ["S", "R"]:
                e = e.shift(-1)

            ens[state] = e

        # Fit modif for each ensemble
        LOGGER.info("Fit to enks ensembles")
        betas = []
        for iens in range(nens):
            if iens%100 == 0:
                LOGGER.info(f".. dealing with ens {iens:3d}/{nens}")
            # Get parameters
            if assim_params == 1:
                # .. from ensemble
                X1e = math.exp(Xa.loc[Xa.state == "X1", \
                                        f"Ens{iens:03d}"].squeeze())
                X2e = math.exp(Xa.loc[Xa.state == "X2", \
                                        f"Ens{iens:03d}"].squeeze())
                Xre = math.exp(Xa.loc[Xa.state == "Xr", \
                                        f"Ens{iens:03d}"].squeeze())
                alphaPe = math.exp(Xa.loc[Xa.state == "alphaP", \
                                        f"Ens{iens:03d}"].squeeze())
                alphaEe = math.exp(Xa.loc[Xa.state == "alphaE", \
                                        f"Ens{iens:03d}"].squeeze())
            else:
                # .. from original calib
                X1e, X2e, Xre, alphaPe, alphaEe, = X1, X2, Xr, 1., 1.

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
            interp_data = gr2m_modif.get_interpolation_variables(esims, \
                                        X1e, X2e, Xre, \
                                        alphaPe, alphaEe, \
                                        useradial=useradial, \
                                        uselinpred=uselinpred, \
                                        useconstraint=useconstraint, \
                                        nodesS=nodesS, \
                                        nodesR=nodesR, \
                                        radfun=radfun, \
                                        radius=radius)

            # Fit interpolation parameters
            pens = gr2m_modif.fit_interpolation(interp_data, \
                                lamP=lamP, lamE=lamE, lamQ=lamQ, \
                                nu=nu, \
                                ical=ical, \
                                usemean=usemean, \
                                rcondS=rcond, \
                                rcondR=rcond, \
                                usebaseline=usebaseline)

            # Store parameters
            betas.append(pens)

        # Compute params fit statistics
        fitstates = pens["params"].keys()
        # .. info data
        pat = "^(use|ra|rc|nod|lam|nu)"
        keys = [k for k in betas[0]["info"].keys() if re.search(pat, k)]
        betas_mean = {
            "info":{k: pens["info"][k] for k in keys}
        }
        # .. fit data
        qq = [0.1, 0.5, 0.9]
        nsefit = [pd.Series([b["info"][f"{s}_diag"]["nse"] \
                                    for s in fitstates],\
                                        index=fitstates) for b in betas]
        nsefit = pd.DataFrame(nsefit)
        for state in fitstates:
            q = nsefit.loc[:, state].quantile(qq).round(2).to_dict()
            q = {f"Q{qq*100:0.0f}%": v for qq, v in q.items()}
            betas_mean["info"][f"{state}_nsefit_stats"] = q

        # Compute params estimator (mean)
        mparams = {s: pens["params"][s]*0. for s in fitstates}
        gparams = np.zeros(5)
        for iens in range(nens):
            for s in fitstates:
                # .. modif params
                mparams[s] += betas[iens]["params"][s]

            # .. gr2m params averaged in log space
            gparams += np.log(betas[iens]["GR2M"])

        for s in fitstates:
             mparams[s] /= nens

        betas_mean["params"] = mparams

        model = gr2m_modif.GR2MMODIF()
        gparams = np.clip(np.exp(gparams/nens), \
                        model.params.mins, \
                        model.params.maxs)

        if assim_params == 0:
            # .. check we haven't changed the parameters
            # .. if they are not assimilated
            assert np.isclose(gparams[0], X1)
            assert np.isclose(gparams[1], X2)
            assert np.isclose(gparams[2], Xr)
            assert np.isclose(gparams[3], 1.)
            assert np.isclose(gparams[4], 1.)

        betas_mean["GR2M"] = gparams.tolist()

        # .. store GR2M modif params (could be different from calib)
        for iparam, pname in enumerate(["X1", "X2", "Xr", "alphaP", "alphaE"]):
            meta[f"PARAM_GR2M-MODIF_{pname}"] = gparams[iparam]

        LOGGER.info("Run sims")
        model.allocate(inputs_val, model.noutputsmax)
        model.set_interp_params(betas_mean)
        assert np.allclose(model.params.values, gparams)
        assert model.get_useradial() == useradial
        assert model.get_uselinpred() == uselinpred
        assert model.get_useconstraint() == useconstraint

        model.initialise_fromdata()
        model.run()
        msims = model.to_dataframe(include_inputs=True, index=mthly.index)
        cc = [cn for cn in msims.columns if not re.search("delta", cn)]
        msims = msims.loc[:, cc]
        msims.columns = [cn if cn in ["Rain", "PET"] else f"MODIF_{cn}" \
                                for cn in msims.columns]

        # .. modif model with not modification to prod
        model_bp = gr2m_modif.GR2MMODIF()
        model_bp.allocate(inputs_val, \
                    noutputs=model_bp.noutputsmax)
        model_bp.set_interp_params(betas_mean)
        # .. this technique works only when using baseline !
        # .. set modif params for S and P3 to 0.
        assert model_bp.get_usebaseline() == 1
        cfg = model_bp.config.to_series().copy()
        cfg[cfg.filter(regex="^(S|P3)").index] = 0
        model_bp.config.values = cfg.values

        model_bp.initialise_fromdata()
        model_bp.run()
        msims_bp = model_bp.to_dataframe(index=mthly.index)

        for  cn in ["Q", "AE", "P3", "R", "S"]:
            v_bp = msims_bp.loc[:, cn]
            if cn in ["AE", "P3", "S"]:
                # We are not changing production function compared to
                # GR2M (not true if we assimilate params though)%
                if assim_params == 0:
                    ck = np.allclose(v_bp, gsims_val.loc[:, cn], \
                                    atol=1e-2, rtol=1e-2)
                    if not ck:
                        errmess = f"msims_bp variable {cn} not matching!"
                        LOGGER.error(errmess)
                        if debug:
                            raise ValueError(errmess)
            else:
                msims.loc[:, f"MODIFBASELINEPROD_{cn}"] = v_bp

        # .. gr2m sims
        for cn in model.outputs_names:
            if re.search("^P$|^E$|nocheck$|update$|baseline$", cn):
                continue
            msims.loc[:, f"GR2M_{cn}"] = gsims_val.loc[:, cn]

        # ... post-processed GR2M
        msims.loc[:, f"PP_Q"] = gpsims_val.Q

        # ... using DA fit GR2M parameters
        grda.params.values = gparams
        grda.initialise_fromdata()
        grda.run()
        msims.loc[:, f"DAPARAMS_Q"] = grda.outputs[:, 0]

        # .. wapaba sims
        msims.loc[:, "WAPABA_Q"] = wsims_val.Q.values
        msims.loc[:, "WAPABA_AE"] = wsims_val.ET.values
        msims.loc[:, "WAPABA_P3"] = wsims_val.Y.values
        # .. Qobs
        msims.loc[:, "Qobs"] = Qobs

        # .. add cal/val periods
        idxcal = calp.active.select_index(msims.index)
        msims.loc[:, "ISCAL"] = 0
        msims.loc[idxcal, "ISCAL"] = 1
        idxval = valp.active.select_index(msims.index)
        msims.loc[:, "ISVAL"] = 0
        msims.loc[idxval, "ISVAL"] = 1

        # Save
        comments = {"comment": "Enks fit results"}
        comments.update(meta)
        comments["config_usemean"] = usemean
        comments["config_rcond"] = rcond
        comments["config_radius"] = radius
        comments["config_radfun"] = radfun
        comments["config_usebaseline"] = usebaseline
        comments["config_uselinpred"] = uselinpred
        comments["config_useconstraint"] = useconstraint

        # Quick verif
        iok = msims.Qobs.notnull() & (msims.ISVAL==1)
        qo = msims.Qobs[iok]
        qsm = msims.MODIF_Q[iok]
        nms = metrics.nse(qo, qsm)
        nlms = metrics.nse(np.log(1+qo), np.log(1+qsm))
        nrms = metrics.nse(1-1/(1+qo), 1-1/(1+qsm))
        comments["METRIC_VAL_NSE_MODIF"] = nms
        comments["METRIC_VAL_NSELOG_MODIF"] = nlms
        comments["METRIC_VAL_NSERECIP_MODIF"] = nrms
        LOGGER.info(f".. NSE[MODIF] : {nms:5.2f}{nlms:5.2f}{nrms:5.2f}")

        qsg = msims.GR2M_Q[iok]
        nmg = metrics.nse(qo, qsg)
        nlmg = metrics.nse(np.log(1+qo), np.log(1+qsg))
        nrmg = metrics.nse(1-1/(1+qo), 1-1/(1+qsg))
        #comments["METRIC_VAL_NSE_GR2M"] = nmg
        #comments["METRIC_VAL_NSELOG_GR2M"] = nlmg
        #comments["METRIC_VAL_NSERECIP_GR2M"] = nrmg
        LOGGER.info(f".. NSE[ GR2M] : {nmg:5.2f}{nlmg:5.2f}{nrmg:5.2f}")

        for state in assim_states:
            n = betas_mean["info"][f"{state}_nsefit_stats"]["Q50%"]
            comments[f"METRIC_CAL_NSEMEDIAN-REGFIT_{state}"] = n

        fn = f"enksfit_sims_{siteid}_{calperiod}_{optid}_v{version}.csv"
        fs = fout / fn
        csv.write_csv(msims, fs, comments, \
                source_file, write_index=True)

        # Store params
        betas_mean["info"]["siteid"] = siteid
        betas_mean["info"]["calperiod"] = calperiod
        betas_mean["info"]["optid"] = optid

        for n in ["nodesS", "nodesR"]+assim_states:
            if n.startswith("nodes"):
                betas_mean["info"][n] = betas_mean["info"][n].tolist()
            else:
                betas_mean["params"][n] = betas_mean["params"][n].to_dict()

        fn = f"enksfit_params_{siteid}_{calperiod}_{optid}_v{version}.json"
        fp = fout / fn
        with fp.open("w") as fo:
            json.dump(betas_mean, fo, indent=4)



LOGGER.info("Process completed")

