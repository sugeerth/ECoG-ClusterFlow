import numpy as np
from scipy import sparse as sp
from gurobipy import *

def gurobi_conductance(B, d, m, n, kappa):
       
    model = Model("mip1")
    
    sol = []
    for j in xrange(n):
        sol.append(model.addVar(vtype=GRB.BINARY, obj=0))
        
    for j in xrange(n,n + m):
        sol.append(model.addVar(lb=0, ub=1, obj=1))

    for j in xrange(n + m,n + 2*m):
        sol.append(model.addVar(lb=0, ub=1, obj=1))
    
    model.update()
    
    for i in xrange(m):
        start = B.indptr[i]
        end   = B.indptr[i+1]
        variables = [sol[j] for j in B.indices[start:end]]
        variables.append(sol[n + i])
        variables.append(sol[n + i + m])
        coeff = np.array([1,-1,1, -1])
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=0)

    expr = gurobipy.LinExpr(d, sol[0:n])
    model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=kappa)
    
    model.update()

    model.ModelSense = 1
    model.setParam('OutputFlag', True )
    model.setParam('MIPFocus',1)
    model.setParam('TimeLimit',30)
    model.optimize()
    
    for i in range(n):
        sol[i] = sol[i].x
    
    sol = np.asarray(sol[0:n], dtype = 'Int64')
    
    sol_nnz = np.nonzero(sol)
    S = sol_nnz[0]
    
    return S, sol