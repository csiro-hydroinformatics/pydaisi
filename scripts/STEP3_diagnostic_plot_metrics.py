#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2023-10-25 Wed 03:38 PM
## Comment : DAISI STEP 3
##           Plot diagnostic metrics
##
## ------------------------------

import sys, os, re, json, math
import argparse
from pathlib import Path
from string import ascii_letters as letters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import boxplot, putils

from pydaisi import daisi_perf

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="DAISI STEP 3 - diagnostic plots", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-fo", "--folder_output", help="Output folder", \
                    type=str, default=None)
args = parser.parse_args()

folder_output = args.folder_output
if not folder_output is None:
    folder_output = Path(folder_output)
    assert folder_output.exists()

# Objective functions
objfun_names = ["kge", "bc02"]

# .. benchmark objective function (to compare with other calib of GR2M)
objfun_bench_names = ["bc02", "kge"]

# Selection of metrics
metric_names = [
    "VAL_KGE", \
    "VAL_NSE", \
    "VAL_NSERECIP", \
    "VAL_NSELOG", \
    "VAL_ABSBIAS", \
    "VAL_ABSFDCFIT100", \
    "VAL_PMR5Y", \
    "VAL_SPLITKGE", \
    "VAL_ELASTrelRAIN", \
]

# Plotting options
ncols = 2
putils.set_mpl(font_size=15)
ax_width = 7
ax_height = 4

# Colors for different models
colgr2m = "tab:green"
coldaisi = "tab:blue"
coldefault = "0.5"

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "STEP3_diagnostic"
if not folder_output is None:
    fout = folder_output / "STEP3_diagnostic"

assert fout.exists()

fimg = fout / "images"
fimg.mkdir(exist_ok=True)

fmetrics = fout / "metrics"
assert fmetrics.exists()

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
fmet = fmetrics.parent / "daisi_metrics.csv"
fz = fmet.parent / f"{fmet.stem}.zip"
if not fz.exists():
    metrics = []
    for f in fmetrics.glob("*.zip"):
        df, _ = csv.read_csv(f)
        metrics.append(df)

    metrics = pd.concat(metrics)
    csv.write_csv(metrics, fmet, "Concatenation of DAISI metrics", \
                    source_file)
else:
    metrics, _ = csv.read_csv(fz)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Plot design
nm = len(metric_names)
nrows = nm//ncols if nm%ncols==0 else nm//ncols+1
mosaic = [[f"{metric_names[ir*ncols+ic]}" if ir*ncols+ic<nm else "." \
                            for ic in range(ncols)]\
                                for ir in range(nrows)]

for objfun_name in objfun_names:
    # Initialise plots
    plt.close("all")
    fig = plt.figure(figsize=(ncols*ax_width, nrows*ax_height), \
                            layout="constrained")
    kw = dict(hspace=0.1)
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
    for iplot, (metric_name, ax) in enumerate(axs.items()):
        cc = [f"METRIC_{metric_name}_GR2M", \
                f"METRIC_{metric_name}_GR2MUPDATE"]

        # Get data corresponding to objective function
        idx = metrics.INFO_OBJFUN == objfun_name
        df = metrics.loc[idx, cc]

        gr2m_name = f"GR2M-{objfun_name}"
        df.columns = [gr2m_name, "DAISI"]

        # Get data corresponding to benchmark objective function
        objfun_bench_name = objfun_bench_names[objfun_names.index(objfun_name)]
        idx_bench = metrics.INFO_OBJFUN == objfun_bench_name
        se = metrics.loc[idx_bench, cc[0]]
        gr2m_bench_name = f"GR2M-{objfun_bench_name}"
        df.loc[:, gr2m_bench_name] = se.values

        df = df.loc[:, [gr2m_name, gr2m_bench_name, "DAISI"]]

        # plot boxplot
        bp = boxplot.Boxplot(df)
        bp.median.show_text = True
        bp.draw(ax)

        # Alter colors
        for cn in bp.stats.columns:
            if cn == gr2m_name:
                col = colgr2m
            elif cn == "DAISI":
                col = coldaisi
            else:
                col = coldefault

            for en, e in bp.elements[cn].items():
                if en == "box":
                    e.set_edgecolor(col)
                else:
                    e.set_color(col)

        for y in bp.stats.loc[["25.0%", "50.0%", "75.0%"], gr2m_name]:
            putils.line(ax, 1, 0, 0, y, "--", color=colgr2m, lw=0.8)

        m = re.sub("VAL_", "", metric_name)
        title = f"({letters[iplot]}) {daisi_perf.get_metricname(m, True)}"
        ax.set_title(title)


    fp = fimg / f"metrics_{objfun_name}.png"
    fig.savefig(fp)

LOGGER.info("Process completed")

