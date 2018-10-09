#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:31:00 2018

@author: abedghanbari
"""

# build cython
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('lam_cython.pyx'),include_dirs=[numpy.get_include()])
