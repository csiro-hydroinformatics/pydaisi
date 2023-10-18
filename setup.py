#!/usr/bin/env python
""" Setup script for pynonstat"""

import os
from pathlib import Path

from setuptools import setup, Extension, find_packages

import versioneer

cmdclass = versioneer.get_cmdclass()

setup(
    name="pydamsi",
    version=versioneer.get_version(),
    cmdclass = cmdclass,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.8.1",
        "pandas>=0.12.1",
        "hydrodiy",
        "pygme"
    ],
    package_data = {
        #"package": [
            #"template/*",
        #]
    },

    # Metadata
    author="Julien Lerat, CSIRO Environment"
)
