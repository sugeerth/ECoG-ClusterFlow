import numpy as np
from scipy import sparse as sp
# For CVX solver
from cvxpy import *

def spectral_improve_cvxpy(B, ei, ej, d, ref_nodes, flow_ref, kappa = 0 , max_iter = 20):
        
    vol_G = sum(d)
    
    size_B = B.shape
    n      = size_B[1]
    m      = size_B[0]
    all_idx = np.asarray(np.arange(n))
    
    S   = flow_ref
    S_c = diff(all_idx,S)
    
    vol_S   = sum(d[S])
    vol_S_c = sum(d[S_c])
    
    vol_ref   = sum(d[ref_nodes])
    
    ref_c = diff(all_idx,ref_nodes)  
    
    vol_ref_c = sum(d[ref_c])
    
    if vol_S > vol_G/2:
        print("vol(ref_nodes) < vol(G)")

    sol = np.zeros((n,1))
    sol[S] = 1  
    
    S_and_Ref   = list(set(S.tolist()) & set(ref_nodes.tolist()))
    S_and_Ref_c = list(set(S.tolist()) & set(ref_c.tolist()))
        
    vol_S_and_Ref   = sum(d[S_and_Ref])
    vol_S_and_Ref_c = sum(d[S_and_Ref_c])
    
    cut_S = B.dot(sol)
    cut_S = np.linalg.norm(cut_S,1)
    
    f_S = (vol_ref/vol_ref_c)*np.exp(kappa)
        
    D_S = vol_S_and_Ref - vol_S_and_Ref_c*f_S
        
    Q_S = cut_S/D_S    
    
    print(Q_S)
   
    d = np.array(d[:,0])
    d = d[:,0]

    weights_s = np.multiply(d[S],Q_S)
    weights_t = np.multiply(d[S_c],Q_S*f_S)
        
    edge_w_aug   = weights_for_gurobi(m, n, len(S), weights_s, len(S_c), weights_t)
    edge_w_aug_Q = sp.diags(edge_w_aug, 0)
    
    edge_w_aug_sqrt   = np.zeros(len(edge_w_aug))
    for i in xrange(len(edge_w_aug)):
        edge_w_aug_sqrt[i] = np.sqrt(edge_w_aug[i])
    edge_w_aug_Q_sqrt = sp.diags(edge_w_aug_sqrt, 0)
    
    data2  = np.append(np.ones((m+n,1)),-np.ones((m+n,1)))
    idx_i2 = np.append(np.array(xrange(len(S))),np.array(xrange(len(S),m+len(S))))
    idx_i2 = np.append(idx_i2,np.array(xrange(len(S)+m,m+n)))
    idx_i2 = np.append(idx_i2,np.array(xrange(len(S))))
    idx_i2 = np.append(idx_i2,np.array(xrange(len(S),m+len(S))))
    idx_i2 = np.append(idx_i2,np.array(xrange(len(S)+m,m+n)))
    idx_j2 = np.append(np.zeros(len(S)),ei+1)
    idx_j2 = np.append(idx_j2,S_c+1)
    idx_j2 = np.append(idx_j2,S+1)
    idx_j2 = np.append(idx_j2,ej+1)
    idx_j2 = np.append(idx_j2,(n+1)*np.ones(len(S_c)))
    B2     = sp.csc_matrix((data2, (idx_i2, idx_j2)), shape=(m+n, n+2))
    
    x = Variable(n+2)
    y = Variable(m+n)
    u = Variable(m+n)
#    objective = Minimize(norm(u,1))
    objective = Minimize(norm(u,2))
    
#    constraints = [x[0]==1, x[n+1]==0, 0 <= x, x <= 1, B2*x - y == 0, u - edge_w_aug_Q*y == 0]
    constraints = [x[0]==1, x[n+1]==0, B2*x - y == 0, u - edge_w_aug_Q_sqrt*y == 0]
    prob = Problem(objective, constraints)

    result = prob.solve(solver='SCS', verbose=True)
#    result = prob.solve(solver='ECOS', verbose=True)

    sol = x.value
        
    return sol[1:n+1]

def weights_for_gurobi(m, n, n_from_s, weights_s, n_to_t, weights_t):

    rows = m + n_from_s + n_to_t
    
    edge_w_aug = np.zeros(rows)
    
    edge_w_aug[0:n_from_s] = weights_s
    edge_w_aug[n_from_s:n_from_s + m] = 1
    edge_w_aug[m + n_from_s:rows] = weights_t
    
    return edge_w_aug
    
def diff(a, b):
        b = set(b)
        return np.asarray([aa for aa in a if aa not in b], dtype = 'int64')