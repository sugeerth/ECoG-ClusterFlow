#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.math cimport abs
from libc.math cimport sqrt


def l1_distance_matrix(int n, double[:,:] d_mat, double[:,:] emb):

    cdef int i, j, k

    for i in range(n):
        print(i)
        for j in range(i,n):
            for k in range(n):
                d_mat[i,j] = d_mat[i,j] + abs(emb[k,i] - emb[k,j])
                
    return d_mat