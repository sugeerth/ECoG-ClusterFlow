import numpy as np
from scipy import sparse as sp

def ista(ref_nodes, A, alpha = 0.15, rho = 1.0e-3, epsilon = 1.0e-1, max_iter = 10000):
    
    size_A = A.shape
    n      = size_A[0]
    S      = ref_nodes
    S_and_neigh = S
    
    len_S = len(S)
    if len_S == 0:
#        print("The set of reference nodes is empty")
        return np.zeros((n, 1))
    
    grad   = np.zeros((n, 1))
    q      = np.zeros((n, 1))

    d_sqrt_S = np.sqrt(A[S,:].sum(axis=1))
    dn_sqrt_S = 1/d_sqrt_S
    
    grad[S] = -(alpha/len_S)*dn_sqrt_S
    
    grad_des = q[S] - grad[S]
       
    thres_vec = rho*alpha*d_sqrt_S
    
    S_filter = np.where(grad_des >= thres_vec)
    S_filter = S_filter[0]
    S = S[S_filter]
    
    if len(S) == 0:
#        print("Parameter rho is too large.")
        return np.zeros((n, 1))
    
    d_sqrt_S = np.sqrt(A[S,:].sum(axis=1))
    dn_sqrt_S = 1/d_sqrt_S
    
    scale_grad = np.multiply(grad[S],-dn_sqrt_S)
        
    max_sc_grad = max(scale_grad)
        
    iter = 1    
        
    while (max_sc_grad > rho*alpha*(1+epsilon) and iter <= max_iter):
        
        direction = -(grad[S] + rho*alpha*d_sqrt_S)
        
        q[S] = q[S] + direction
        
        grad = grad + mat_vec_with_Q(A,S,dn_sqrt_S,alpha,direction)
    
        S_and_neigh = grad.nonzero()
        S_and_neigh = S_and_neigh[0]
        
        grad_des = q[S_and_neigh] - grad[S_and_neigh]
    
        d_sqrt_S_and_neigh = np.sqrt(A[S_and_neigh,:].sum(axis=1))
       
        thres_vec = rho*alpha*d_sqrt_S_and_neigh
    
        S = np.where(grad_des >= thres_vec)
        S = S[0]
        S = S_and_neigh[S]
        
        d_sqrt_S = np.sqrt(A[S,:].sum(axis=1))
        dn_sqrt_S = 1/d_sqrt_S
        
        scale_grad = np.multiply(grad[S],-dn_sqrt_S)
        
        max_sc_grad = max(scale_grad)
        
        iter = iter + 1
    
    nnz_q = np.nonzero(q)
    nnz_q = nnz_q[0]
            
    d_sqrt_nnz_q = np.sqrt(A[nnz_q,:].sum(axis=1))
            
    p = np.array(q)
    p[nnz_q] = np.multiply(q[nnz_q],d_sqrt_nnz_q)
        
#    print "Terminated at iteration %d." % iter
#    print "Termination criterion %f." % max_sc_grad
#    print "sum(p) %f." % sum(p)
#    print "sum(q) %f." % sum(q)
            
    return p
    
def mat_vec_with_Q(A,S,dn_sqrt_S,alpha,x):
    
    y = np.multiply(x, dn_sqrt_S)
    y = A[:,S].dot(y)
    
    nnz_y = np.nonzero(y)
    nnz_y = nnz_y[0]
    
    dn_sqrt_nnz_y = 1/np.sqrt(A[nnz_y,:].sum(axis=1))
    
    y[nnz_y] = np.multiply(y[nnz_y],dn_sqrt_nnz_y)
    y = np.multiply(y,-(1-alpha)/2)
    y[S] = y[S] + np.dot(x,(1+alpha)/2)
    
    return y
    
    
    
    
    