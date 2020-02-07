#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
import subprocess as sp
import sys
from distutils.errors import DistutilsError
from distutils.sysconfig import get_config_var, get_config_vars

import numpy as np
import setuptools
import versioneer
from numpy.distutils.core import setup, Extension, build_ext, build_src

build_ext = build_ext.build_ext
build_src = build_src.build_src

setup(
    author="Dongwon 'DW' Han",
    author_email='dwhan89@gmail.com',
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Intended Audience :: Mostly Me',
        'License :: Apache LICENSE-2.0',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Cosmimikyu",
    install_requires=["torch",
                      "numpy >= 1.10",
                      "pixell",
                      "scipy",
                      "mpi4py",
                      "pandas"],
    license="Apache LICENSE-2.0",
    keywords='Cosmimikyu',
    name='Cosmimikyu',
    packages=['Cosmimikyu'],
    include_package_data=True,
    package_data={'cosmimikyu': ['data/*']},
    url='https://github.com/dwhan89/cosmimikyu',
)

print('\n[setup.py request was successful.]')
