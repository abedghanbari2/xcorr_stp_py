#cython_loop.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)

cpdef stp_model_cython(double[:] isi, double[:] theta):
    cdef:
        int L = isi.shape[0], i, j
        double[:] psp = np.zeros(L, dtype=np.float64)
        double tauD = theta[0]
        double tauF = theta[1]
        double U = theta[2]
        double f = theta[3]
        double tauInteg = theta[4]
    R = 1
    u = U
    psp[0] = U
    for i in range(L-1):
        R = 1 - (1 - R * (1 - u)) * exp(-isi[i] / tauD)
        u = U + (u + f * (1 - u) - U) * exp(-isi[i] / tauF)
        psp[i + 1] = psp[i] * exp(-isi[i] / tauInteg) + R * u
    return psp

