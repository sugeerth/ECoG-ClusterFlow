import numpy as np
from scipy import sparse as sp

def MOV(A, d, ref_nodes, gamma, case = 1):
    
    n = A.shape[0]
    
    D = sp.spdiags(d.transpose(), 0, n, n)
    
    L = D - A
    
    L_reg = L - D.multiply(gamma)
    
    s = np.zeros((n,1))
    
    if case == 0:
        for i in ref_nodes:
            s[i] = 1/d[i]
    else:
        s[ref_nodes] = 1
    #othogonalizes s relative to degree sequence
    s = s - np.multiply(d,(np.dot(s.transpose(),d)/np.dot(d.transpose(),d)))
    s = np.multiply(s,1/np.sqrt(np.dot(s.transpose(),np.multiply(s,d))))
    b = D.dot(s)
    
    x, info = sp.linalg.minres(L_reg, b, tol=1e-05, maxiter=3000)
    
    return x

    
    