#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def acl(int n, int ref_node, double[:] d, double[:] A_data, int[:] A_indices, int[:] A_indptr, double alpha = 0.15, double rho = 1.0e-3, max_iter = 10000):

    cdef int idx
    cdef double max_idx
    
    cdef double[:] r = np.zeros(n)
    cdef double[:] p = np.zeros(n)
    cdef double direction
    
    cdef list nodes = []
    
    nodes.append(ref_node)
    r[ref_node] = 1
    
    iter = 0        
         
    while len(nodes) > 0 and iter <= max_iter:
        
        idx = 0
        max_idx = 0
        for u in range(len(nodes)):
            if max_idx < r[nodes[u]]:
                idx = nodes[u]
                
#        idx = nodes[0]
        
        direction = r[idx]
        p[idx] = p[idx] + alpha*direction
        r[idx] = ((1-alpha)/2)*direction
        
        if r[idx] < rho*d[idx]:
            del nodes[u]  
        
        for u in range(A_indptr[idx],A_indptr[idx+1]):
            j = A_indices[u]
            r[j] = r[j] + ((1-alpha)/2)*(direction/d[idx])*A_data[u]
            if r[j] >= rho*d[j]:
                nodes.append(j)       
            
        iter = iter + 1    
    return p, r
        
    
    
    