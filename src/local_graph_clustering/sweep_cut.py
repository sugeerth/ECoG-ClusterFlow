import numpy as np
from scipy import sparse as sp

def sweep_cut_conductance_degree_normalized(B, d, p):
    
    n = len(p)
    m = B.shape[0]
    
    nnz_ct = np.count_nonzero(p) 
    nnz_idx = p.nonzero()[0]
    
    sc_p = np.zeros((nnz_ct,1))
    for i in xrange(nnz_ct):
        sc_p[i] = p[nnz_idx[i]]/d[nnz_idx[i]]
            
    srt_idx = np.argsort((-sc_p).transpose())
    srt_idx = srt_idx[0]
    
    best_conductance = np.inf
    best_support = []
    
    vol_S = 0
    Bsol  = sp.lil_matrix((m,1))
    
    vol_G = sum(d)
    
    # If the embedding is non-negative. then there is 
    # no reason to include zeros in sweep cut.
    size_loop = nnz_ct
    if size_loop == n:
        size_loop = n - 1
        
    for i in xrange(size_loop):
  
        Bsol = Bsol + B[:,nnz_idx[srt_idx[i]]]
        cut_S = np.linalg.norm(Bsol.data, 1)
        
        vol_S = vol_S + d[nnz_idx[srt_idx[i]]]
        vol_S_c = vol_G - vol_S
        
        denominator = min(vol_S,vol_S_c)
        
        cond = cut_S/denominator
        
        if cond < best_conductance:
            best_conductance = cond
            best_support = nnz_idx[srt_idx[0:i+1]]
            
    return best_support, best_conductance

def sweep_cut_conductance_degree_sqrt_normalized_map(B, d, p, mapp):
    
    n = len(p)
    m = B.shape[0]
    
    nnz_ct = np.count_nonzero(p) 
    nnz_idx = p.nonzero()[0]
    
    sc_p = np.zeros((nnz_ct,1))
    for i in xrange(nnz_ct):
        sc_p[i] = p[nnz_idx[i]]/np.sqrt(d[mapp[nnz_idx[i]]])
            
    srt_idx = np.argsort((sc_p).transpose())
    srt_idx = srt_idx[0]
    
    best_conductance = np.inf
    best_support = []
    
    vol_S = 0
    Bsol  = sp.lil_matrix((m,1))
    
    vol_G = sum(d)
    
    # If the embedding is non-negative. then there is 
    # no reason to include zeros in sweep cut.
    size_loop = nnz_ct
    if size_loop == n:
        size_loop = n - 1
        
    for i in xrange(size_loop):
  
        Bsol = Bsol + B[:,mapp[nnz_idx[srt_idx[i]]]]
        cut_S = np.linalg.norm(Bsol.data, 1)
        
        vol_S = vol_S + d[mapp[nnz_idx[srt_idx[i]]]]
        vol_S_c = vol_G - vol_S
        
        denominator = min(vol_S,vol_S_c)
        
        cond = cut_S/denominator
        
        if cond < best_conductance:
            best_conductance = cond
            best_support = mapp[nnz_idx[srt_idx[0:i+1]]]
            
    return best_support, best_conductance

def sweep_cut_conductance(B, d, p):
    
    n = len(p)
    m = B.shape[0]
    
    nnz_ct = np.count_nonzero(p) 
    nnz_idx = p.nonzero()[0]
    
    sc_p = np.zeros((nnz_ct,1))
    for i in xrange(nnz_ct):
        sc_p[i] = p[nnz_idx[i]]
            
    srt_idx = np.argsort((-sc_p).transpose())
    srt_idx = srt_idx[0]
    
    best_conductance = np.inf
    best_support = []
    
    vol_S = 0
    Bsol  = sp.lil_matrix((m,1))
    
    vol_G = sum(d)
    
    # If the embedding is non-negative. then there is 
    # no reason to include zeros in sweep cut.
    size_loop = nnz_ct
    if size_loop == n:
        size_loop = n - 1
        
    for i in xrange(size_loop):
  
        Bsol = Bsol + B[:,nnz_idx[srt_idx[i]]]
        cut_S = np.linalg.norm(Bsol.data, 1)
        
        vol_S = vol_S + d[nnz_idx[srt_idx[i]]]
        vol_S_c = vol_G - vol_S
        
        denominator = min(vol_S,vol_S_c)
        
        cond = cut_S/denominator
        
        if cond < best_conductance:
            best_conductance = cond
            best_support = nnz_idx[srt_idx[0:i+1]]
            
    return best_support, best_conductance