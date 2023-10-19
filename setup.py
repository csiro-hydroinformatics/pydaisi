#!/usr/bin/env python
""" Setup script for pynonstat"""

import os
from pathlib import Path
import subprocess
import numpy
import pygme

from setuptools import setup, Extension, find_packages

import distutils.cmd
import distutils.log

from Cython.Distutils import build_ext

import versioneer

cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext

pygme_include_dir = Path(pygme.__file__).parent / "models"

ext_modules = [
    Extension(
        name="c_pydamsi",
        sources=[
            "pydamsi/c_pydamsi.pyx",
            "pydamsi/c_gr2m_update.c",\
            "pydamsi/c_damsi_utils.c",\
            str(pygme_include_dir / "c_utils.c"),
        ],
        include_dirs=[\
            numpy.get_include(), \
            str(pygme_include_dir)
        ])
]

setup(
    name="pydamsi",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.8.1",
        "pandas>=0.12.1",
        "hydrodiy",
        "pygme"
    ],
    ext_modules = ext_modules,
    package_data = {
        #"package": [
            #"template/*",
        #]
    },
    # Metadata
    author="Julien Lerat, CSIRO Environment"
)
