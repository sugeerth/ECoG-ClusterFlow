import numpy as np
from scipy import sparse as sp
from gurobipy import *

def MQI(A, d, ref_nodes, max_iter = 20):
    
    n = A.shape[0]
    
    if sum(d[ref_nodes]) > sum(d)/2:
        print("vol(ref_nodes) < vol(G)")     
        
    # Calculate directed incidence.
    Atriu = sp.triu(A)
    ei,ej = Atriu.nonzero()
    Atriu = []; m = len(ei)
    
    data = np.append(np.ones((m,1)),-np.ones((m,1)))
    idx_i = np.append(np.array(xrange(m)),np.array(xrange(m)))
    idx_j = np.append(ei,ej)
    B = sp.csc_matrix((data, (idx_i, idx_j)), shape=(m, n))
    data = []; idx_i = []; idx_j = []; ei = []; ej = []
    
    # Calculate directed incidence sub-matrix.
    A_sub = A.tocsr()[ref_nodes, :].tocsc()[:, ref_nodes]
    Atriu = sp.triu(A_sub)
    ei,ej = Atriu.nonzero()
    Atriu = []; m_sub  = len(ei);
    
    data  = np.append(np.ones((m_sub,1)),-np.ones((m_sub,1)))
    idx_i = np.append(np.array(xrange(m_sub)),np.array(xrange(m_sub)))
    idx_j = np.append(ei,ej)
    B_sub = sp.csc_matrix((data, (idx_i, idx_j)), shape=(m_sub, len(ref_nodes)))
    data  = []; idx_i = []; idx_j = []; ei = []; ej = [];
    
    B_sub = B_sub.tocsr() 
    
    size_B_sub = B_sub.shape
    n_sub      = size_B_sub[1]
    m_sub      = size_B_sub[0]
    
    all_idx_sub = np.asarray(np.arange(n_sub))
    S_sub       = all_idx_sub
    
    size_B = B.shape
    n      = size_B[1]
    m      = size_B[0]
    
    S = ref_nodes
    
    vol_S = sum(d[S])

    sol    = np.zeros((n,1))
    sol[S] = 1
    
    cut_S = B.dot(sol)
    cut_S = np.linalg.norm(cut_S,1)
    #print "CUT %f." % cut_S
    
    d_sub = A_sub.sum(axis=1)
    
    cut_S_old = cut_S + 1
    vol_S_old = 1
    
    #print "Iteration %d, Objective val. %f." % (0, cut_S/vol_S)
    
    iter = 0
    
    while (cut_S/vol_S < cut_S_old/vol_S_old and iter <= max_iter):

        weights_s_sub = np.multiply(d[S],cut_S/vol_S)
        weights_t_sub = np.multiply(d[S] - d_sub,1)
        
        edge_w_aug_sub = weights_for_gurobi(m_sub, len(S_sub), weights_s_sub, weights_t_sub)

        S_sub, sol_sub = gurobi_st_min_cut(B_sub, edge_w_aug_sub, S_sub, all_idx_sub, m_sub, n_sub)
    
        S_old  = S
        S      = ref_nodes[S_sub]
        sol    = np.zeros((n,1))
        sol[S] = 1
        
        # Calculate directed incidence.
        A_sub = A.tocsr()[S, :].tocsc()[:, S]
        Atriu = sp.triu(A_sub)
        ei,ej = Atriu.nonzero()
        Atriu = []; m_sub  = len(ei);
        
        data  = np.append(np.ones((m_sub,1)),-np.ones((m_sub,1)))
        idx_i = np.append(np.array(xrange(m_sub)),np.array(xrange(m_sub)))
        idx_j = np.append(ei,ej)
        B_sub = sp.csc_matrix((data, (idx_i, idx_j)), shape=(m_sub, len(S)))
        data  = []; idx_i = []; idx_j = []; ei = []; ej = [];
    
        B_sub = B_sub.tocsr() 
    
        size_B_sub = B_sub.shape
        n_sub      = size_B_sub[1]
        m_sub      = size_B_sub[0]
    
        all_idx_sub = np.asarray(np.arange(n_sub))
    
        vol_S_old = vol_S
        vol_S = sum(d[S])
    
        cut_S_old = cut_S
        cut_S = B.dot(sol)
        cut_S = np.linalg.norm(cut_S,1)
    
        d_sub = A_sub.sum(axis=1)
            
        if vol_S == 0:
            print "Iteration %d, Objective val. %f." % (iter + 1, cut_S_old/vol_S_old)
            return S_old
        
        if cut_S/vol_S > cut_S_old/vol_S_old:
            print "Iteration %d, Objective val. %f." % (iter + 1, cut_S_old/vol_S_old)
            return S_old
            
        iter = iter + 1
    
        #print "Iteration %d, Objective val. %f." % (iter, cut_S/vol_S)        
        
    return S
        
def gurobi_st_min_cut(B, edge_w_aug, S, all_idx, m, n):
    
    rows = m + len(S) + n
    cols = n + 2 + 2*rows    
    
    model = Model()
    
    sol = []
    for j in xrange(n):
        sol.append(model.addVar(lb=0, ub=1, obj=0))
        
    sol.append(model.addVar(lb=1, ub=1, obj=0))
    sol.append(model.addVar(lb=0, ub=0, obj=0))
    
    for j in xrange(n + 2,n + 2 + rows):
        sol.append(model.addVar(lb=0, ub=1, obj=edge_w_aug[j - (n + 2)]))

    for j in xrange(n + 2 + rows,cols):
        sol.append(model.addVar(lb=0, ub=1, obj=edge_w_aug[j - (n + 2 + rows)]))
        
    model.update()
    
    for i in xrange(m):
        start = B.indptr[i]
        end   = B.indptr[i+1]
        variables = [sol[j] for j in B.indices[start:end]]
        variables.append(sol[n + 2 + i])
        variables.append(sol[n + 2 + rows + i])
        coeff = np.array([1, -1, 1, -1])
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=0)

    for i in xrange(len(S)):
        variables = [sol[S[i]], sol[n], sol[n + 2 + m + i], sol[n + 2 + rows + m + i]]
        coeff     = np.array([1,-1,1,-1])
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=0)
        
    for i in xrange(n):
        variables = [sol[all_idx[i]], sol[n + 1], sol[n + 2 + m + len(S) + i], sol[n + 2 + rows + m + len(S) + i]]
        coeff     = np.array([1,-1,1,-1])
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=0)
        
    model.update()

    model.setParam("Method", 1);
    model.setParam("OutputFlag",0)
    model.ModelSense = 1
    model.optimize()
    
    for i in range(n):
        sol[i] = sol[i].x
    
    sol = np.asarray(sol[0:n], dtype = 'Int64')
    
    
    sol_nnz = np.nonzero(sol)
    S = sol_nnz[0]
    
    return S, sol

def weights_for_gurobi(m, n_from_s, weights_s, weights_t):

    rows = m + 2*n_from_s
    
    edge_w_aug      = sp.lil_matrix((rows, 1), dtype = 'float64')
    edge_w_aug[0:m] = 1
    
    edge_w_aug[m:m + n_from_s]    = weights_s
    edge_w_aug[m + n_from_s:rows] = weights_t
    
    edge_w_aug = edge_w_aug.todense()
    
    return edge_w_aug
    
def diff(a, b):
        b = set(b)
        return np.asarray([aa for aa in a if aa not in b], dtype = 'int64')