import numpy as np
from scipy import sparse as sp

def eig2_L(A, d, tol_eigs = 1.0e-6):
    
    n = A.shape[0]
    
    d_sqrt = np.sqrt(d)
    d_sqrt_neg = np.zeros((n,1))
    
    for i in xrange(n):
        d_sqrt_neg[i] = 1/d_sqrt[i]
    
    D_sqrt_neg = sp.spdiags(d_sqrt_neg.transpose(), 0, n, n)
    
    L = sp.identity(n) - D_sqrt_neg.dot((A.dot(D_sqrt_neg)))
    
    emb_eig_val, emb_eig = sp.linalg.eigs(L, which='SM', k=2, tol = tol_eigs)
    
    return np.real(emb_eig[:,1]), emb_eig_val[1]
    