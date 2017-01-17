#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.math cimport abs
from libc.math cimport sqrt
from libc.math cimport log

#from cython.parallel import prange

def KL_divergence(int n, double[:] u, double[:] v):

    cdef int i, j, k
    cdef double result
    
    result = 0
    
    for i in range(n):
        result = result + u[i]*log((u[i] + 1.0e-10)/(v[i] + 1.0e-10))
                
    return sqrt(result)

def sym_KL_divergence(int n, double[:] u, double[:] v):

    cdef int i, j, k
    cdef double result_uv, result_vu, result
    
    result_uv = 0
    result_vu = 0
    
    for i in range(n):
        result_uv = result_uv + u[i]*log((u[i] + 1.0e-10)/(v[i] + 1.0e-10))
        result_vu = result_vu + v[i]*log((v[i] + 1.0e-10)/(u[i] + 1.0e-10))
        
    result = sqrt(result_uv + result_vu)
                
    return result

def sym_JS_divergence(int n, double[:] u, double[:] v):

    cdef int i, j, k
    cdef double result_um, result_vm, result
    cdef double[:] m = np.zeros(n)
    
    for i in range(n):
        m[i] = (u[i] + v[i])/2
    
    result_um = 0
    result_vm = 0
    
    for i in range(n):
        if u[i] != 0:
            result_um = result_um + u[i]*log((u[i] + 1.0e-10)/(m[i] + 1.0e-10))
        if v[i] != 0:
            result_vm = result_vm + v[i]*log((v[i] + 1.0e-10)/(m[i] + 1.0e-10))
        
    result = sqrt(0.5*result_um + 0.5*result_vm)
                   
    return result

def sym_JS_matrix(int n, double [:,:] emb, int step):
    
    cdef int i, j, n_C_2, idx, temp
    
    n_C_2 = n*(n-1)/2
    
    cdef double[:] Y = np.ones(n_C_2)
    
    for i in range(n_C_2):
        Y[i] = 0
    
    for i in range(n-1):
        print(i)
        if i == 0:
            temp = 0
        else:
            temp = temp + n - i - 1
        
        stepp = min(i+step,n)
        for j in range(i+1,stepp):
            idx = temp + j - 1
            Y[idx] = sym_JS_divergence(n, emb[:,i], emb[:,j])
            
    return Y
    
def sym_JS_matrix_fast(int n, int p, int q, double [:,:] emb, double r):
    
    cdef int i, j, k, l, n_C_2, temp
    cdef double wkl, d, nF, temp1, temp2
    
    n_C_2 = n*(n-1)/2
    
    cdef double[:,:] X
    cdef double[:] data
    
    X = np.zeros([n, 2])
    data = np.zeros(n_C_2)
    
    for i in range(n_C_2):
        data[i] = 1
    
    for i in range(n):
        X[i,0] = 0
        X[i,1] = 0
    
    # Positions
    for i in range(p):
        for j in range(q):
            k = i*q + j
            X[k,0] = i
            X[k,1] = j

    for k in range(n):
        if k%100 == 0:
            print k
        if k == 0:
            temp = 0
        else:
            temp = temp + n - k - 1
            
        for l in range(k+1, n):
            temp1 = abs(X[k,0] - X[l,0])
            temp1 = temp1*temp1
            temp2 = abs(X[k,1] - X[l,1])
            temp2 = temp2*temp2
            d = sqrt(temp1 + temp2)
            if d < r:
                wkl = sym_JS_divergence(n, emb[:,k], emb[:,l])
                idx = temp + l - 1
                data[idx] = wkl 
    return data 
    
def sym_JS_matrix_fast2(int n, int p, int q, double[:] emb_data, int[:] emb_indices, int[:] emb_indptr, double r):
    
    cdef int i, j, k, l, temp
    cdef double wkl, d, nF, temp1, temp2
    cdef long long n_C_2
    
    cdef double result_um, result_vm
    cdef double[:] m = np.zeros(n)
    
    n_C_2 = n*(n-1)/2
    
    cdef double[:,:] X
    cdef double[:] data
    
    X = np.zeros([n, 2])
    data = np.zeros(n_C_2)
    
    for i in range(n_C_2):
        data[i] = 1
    
    for i in range(n):
        X[i,0] = 0
        X[i,1] = 0
    
    # Positions
    for i in range(q):
        for j in range(p):
            k = j + i*p
            X[k,0] = j
            X[k,1] = i

    for k in range(n):
        #if k%100 == 0:
        #    print k
        if k == 0:
            temp = 0
        else:
            temp = temp + n - k - 1
            
        for l in range(k+1, n):
            temp1 = abs(X[k,0] - X[l,0])
            temp1 = temp1*temp1
            temp2 = abs(X[k,1] - X[l,1])
            temp2 = temp2*temp2
            d = sqrt(temp1 + temp2)
            if d < r:
                
                for i in range(emb_indptr[k],emb_indptr[k+1]):
                    j = emb_indices[i]
                    m[j] = emb_data[i]
                    
                for i in range(emb_indptr[l],emb_indptr[l+1]):
                    j = emb_indices[i]
                    m[j] = m[j] + emb_data[i]
                    
                for i in range(n):
                    m[i] = m[i]/2
    
                result_um = 0
                result_vm = 0
    
                for i in range(emb_indptr[k],emb_indptr[k+1]):
                    j = emb_indices[i]
                    if emb_data[i] != 0:
                        result_um = result_um + emb_data[i]*log((emb_data[i] + 1.0e-10)/(m[j] + 1.0e-10))
                
                for i in range(emb_indptr[l],emb_indptr[l+1]):
                    j = emb_indices[i]
                    if emb_data[i] != 0:
                        result_vm = result_vm + emb_data[i]*log((emb_data[i] + 1.0e-10)/(m[j] + 1.0e-10))
        
                idx = temp + l - 1
            
                data[idx] = sqrt(0.5*result_um + 0.5*result_vm)
                
                m = np.zeros(n)
                
    return data     

def sym_Euclidean_matrix_fast2(int n, int p, int q, double[:] emb_data, int[:] emb_indices, int[:] emb_indptr, double r):
    
    cdef int i, j, k, l, temp
    cdef double wkl, d, nF, temp1, temp2
    cdef long long n_C_2
    
    cdef double result
    cdef double[:] m = np.zeros(n)
    
    n_C_2 = n*(n-1)/2
    
    cdef double[:,:] X
    cdef double[:] data
    
    X = np.zeros([n, 2])
    data = np.zeros(n_C_2)
    
    for i in range(n_C_2):
        data[i] = 100
    
    for i in range(n):
        X[i,0] = 0
        X[i,1] = 0
    
    # Positions
    for i in range(q):
        for j in range(p):
            k = j + i*p
            X[k,0] = j
            X[k,1] = i

    for k in range(n):
        #if k%100 == 0:
        #    print k
        if k == 0:
            temp = 0
        else:
            temp = temp + n - k - 1
            
        for l in range(k+1, n):
            temp1 = abs(X[k,0] - X[l,0])
            temp1 = temp1*temp1
            temp2 = abs(X[k,1] - X[l,1])
            temp2 = temp2*temp2
            d = sqrt(temp1 + temp2)
            if d < r:
                
                for i in range(emb_indptr[k],emb_indptr[k+1]):
                    j = emb_indices[i]
                    m[j] = emb_data[i]*emb_data[i]
                    
                for i in range(emb_indptr[l],emb_indptr[l+1]):
                    j = emb_indices[i]
                    m[j] = m[j] - emb_data[i]*emb_data[i]
                    
                result = 0    
                    
                for i in range(n):
                    result = result + m[i]
        
                idx = temp + l - 1
            
                data[idx] = sqrt(result)
                
                m = np.zeros(n)
                
    return data     
