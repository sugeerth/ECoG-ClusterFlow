import numpy as np
from scipy import sparse as sp
# For CVX solver
from cvxpy import *

def spectral_improve(A, d, flow_ref, Q_S):
        
    n = A.shape[0]
    
    D = sp.spdiags(d.transpose(), 0, n, n)
    
    L = Q_S[0,0]*D + (D - A)
    
    rhs = np.zeros(n)
    for i in flow_ref:
        rhs[i] = d[i]
    
    #x = sp.linalg.spsolve(L,rhs)
    x, info = sp.linalg.cg(L,rhs)
    
    return x
    