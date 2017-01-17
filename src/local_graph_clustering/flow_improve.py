import numpy as np
from scipy import sparse as sp
from gurobipy import *

def flow_improve(B, d, ref_nodes, kappa = 0 , max_iter = 20):
    
    B = B.tocsr()
    
    vol_G = sum(d)
    
    size_B = B.shape
    n      = size_B[1]
    m      = size_B[0]
    all_idx = np.asarray(np.arange(n))
    
    S   = ref_nodes
    S_c = diff(all_idx,S)
    
    ref_c = diff(all_idx,ref_nodes)
    
    vol_S   = sum(d[S])
    vol_S_c = sum(d[S_c])
    
    if vol_S > vol_G/2:
        print("vol(ref_nodes) < vol(G)")

    sol = np.zeros((n,1))
    sol[S] = 1
    
    cut_S = B.dot(sol)
    cut_S = np.linalg.norm(cut_S,1)
    #print "CUT %f." % cut_S
    
    f_S = (vol_S/vol_S_c)*np.exp(kappa)
    #print "np.exp(kappa) %f." % np.exp(kappa)
    D_S = vol_S
    
    #print "D_S %f." % D_S
    
    Q_S = cut_S/D_S
    Q_S_old = Q_S + 1
    
    #print "Iteration %d, Objective val. %f." % (0, Q_S)
    
    iter = 0
    
    while (Q_S < Q_S_old and iter <= max_iter):

        weights_s = np.multiply(d[S],Q_S)
        weights_t = np.multiply(d[S_c],Q_S*f_S)
        
        edge_w_aug = weights_for_gurobi(m, n, len(S), weights_s, len(S_c), weights_t)
        
        S_old  = S
        S, sol = gurobi_st_min_cut(B, edge_w_aug, S, S_c, m, n)
        
        S_c_old = S_c
        S_c     = diff(all_idx,S)
        
        S_and_Ref   = list(set(S.tolist()) & set(ref_nodes.tolist()))
        S_and_Ref_c = list(set(S.tolist()) & set(ref_c.tolist()))
        
        vol_S_and_Ref   = sum(d[S_and_Ref])
        vol_S_and_Ref_c = sum(d[S_and_Ref_c])
    
        cut_S = B.dot(sol)
        cut_S = np.linalg.norm(cut_S,1)
        #print "CUT %f." % cut_S
        
        D_S = vol_S_and_Ref - vol_S_and_Ref_c*f_S
        
        #print "D_S %f." % D_S
        
        Q_S_old = Q_S
        
        if D_S == 0:
            print "Iteration %d, Objective val. %f." % (iter + 1, Q_S_old)
            return S_old
        
        Q_S = cut_S/D_S
            
        iter = iter + 1
    
        #print "Iteration %d, Objective val. %f." % (iter, Q_S)        
        
    return S, Q_S
        
def gurobi_st_min_cut(B, edge_w_aug, S, S_c, m, n):
    
    rows = m + n
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
        
    for i in xrange(len(S_c)):
        variables = [sol[S_c[i]], sol[n + 1], sol[n + 2 + m + len(S) + i], sol[n + 2 + rows + m + len(S) + i]]
        coeff     = np.array([1,-1,1,-1])
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=0)
        
    model.update()

    model.setParam("Method", 1);
    model.setParam("OutputFlag", 0);
    model.ModelSense = 1
    model.optimize()
    
    for i in range(n):
        sol[i] = sol[i].x
    
    #print "SUM %f." % sum(sol[0:n]) 
    
    sol = np.asarray(sol[0:n], dtype = 'Int64')
    
    #print "SUM2 %f." % sum(sol)
    
    sol_nnz = np.nonzero(sol)
    S = sol_nnz[0]
    
    return S, sol

def weights_for_gurobi(m, n, n_from_s, weights_s, n_to_t, weights_t):

    rows = m + n_from_s + n_to_t
    
    edge_w_aug      = sp.lil_matrix((rows, 1), dtype = 'float64')
    edge_w_aug[0:m] = 1
    
    edge_w_aug[m:m + n_from_s]    = weights_s
    edge_w_aug[m + n_from_s:rows] = weights_t
    edge_w_aug = edge_w_aug.todense()
    
    return edge_w_aug
    
def diff(a, b):
        b = set(b)
        return np.asarray([aa for aa in a if aa not in b], dtype = 'int64')