#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:58:28 2018

@author: abedghanbari
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)


cpdef lambda_fun_cython(double[:] beta, double[:,:] X, double[:] psp_scaled, double[:] synapse, double[:,:] mY):
    cdef:
        int N = X.shape[0], M = X.shape[1], L = mY.shape[1], i, j
        double[:] bX = np.zeros(N, dtype=np.float)
        double[:,:] lam = np.zeros([N,L], dtype=np.float)
    for i in range(N):
        bX[i] = 0
        for j in range(M):
            bX[i] = bX[i] + beta[j]*X[i,j]
        for j in range(L):
            lam[i,j] = mY[i,j]*1/(1+exp(-(bX[i]+psp_scaled[i]*synapse[100+j])))
    return lam
