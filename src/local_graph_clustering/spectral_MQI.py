import numpy as np
from scipy import sparse as sp
from scipy import linalg as sp_linalg

def spectral_MQI(A, d, ref_nodes):
    
    A_sub = A.tocsr()[ref_nodes, :].tocsc()[:, ref_nodes]
    
    n = A_sub.shape[0]
    
    d_sqrt = np.sqrt(d[ref_nodes])
    d_sqrt_neg = np.zeros((n,1))
    
    for i in xrange(n):
        d_sqrt_neg[i] = 1/d_sqrt[i]
    
    D_sqrt_neg = sp.spdiags(d_sqrt_neg.transpose(), 0, n, n)
    
    L_sub = sp.identity(n) - D_sqrt_neg.dot((A_sub.dot(D_sqrt_neg)))
    
    emb_eig_val, emb_eig = sp.linalg.eigs(L_sub, which='SM', k=1, tol=1.0e-9)
    
    #nL_aug = sp.lil_matrix((n+1,n+1))
    #nL_aug[0:n,0:n] = L_sub
    #nL_aug[n,0:n] = -d_sqrt.transpose()
    #nL_aug[0:n,n] = -d_sqrt
    #nL_aug[n,n] = 0    
    
    #U,S,V = sp_linalg.svd(nL_aug.todense())
    
    #return U[0:n,n]
    return emb_eig