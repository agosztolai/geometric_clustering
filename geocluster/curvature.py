#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import ot
from tqdm import tqdm

'''
=============================================================================
Functions for computing the curvature
=============================================================================
'''

# compute all neighbourhood densities
def mx_comp(L, dt, mx):
    """ compute matrix exponential """

    return sc.sparse.linalg.expm_multiply(-dt*L, mx)

# compute curvature for an edge ij
def K_ij(mxs, dist, lamb, cutoff, with_weights, edges, e):
    #print("step "+ str(e))

    # get the edge/nodes ids
    edge = edges[e]
    i = edge[0]
    j = edge[1]

    #get the measures
    mx = mxs[i]
    my = mxs[j]
    
    #set reduce the sized with cutoffs
    Nx = np.where(mx >= (1. - cutoff) * np.max(mx))[0]
    Ny = np.where(my >= (1. - cutoff) * np.max(my))[0]

    dNxNy = dist[np.ix_(Nx, Ny)]

    mx = mx[Nx]
    my = my[Ny]

    mx /=mx.sum()
    my /=my.sum()

    #compute K
    if lamb != 0: #entropy regularized OT
        W = ot.sinkhorn2(mx, my, dNxNy, lamb)

    elif lamb == 0: #classical sparse OT
        W = ot.emd2(mx, my, dNxNy)
        
    if with_weights:
        K = dist[i, j] - W
    else:
        K = 1. - W / dist[i, j]  
     
    return K


def K_all(mx_all, dist, lamb, G, with_weights=False):     

    dist = dist.astype(float)
    
    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in tqdm(x):
        ind = [y[1] for y in G.edges if y[0] == i]              

        W = ot.sinkhorn(mx_all[:,i].tolist(), mx_all[:,ind].tolist(), dist.tolist(), lamb)    
        if with_weights:
            Kt = np.append(Kt, dist[i][ind] - W)
        else:
            Kt = np.append(Kt, 1. - W/dist[i][ind])
        
    return Kt


def K_all_gpu(mx_all, dist, lamb, G, with_weights=False):   
    import ot.gpu    
       
    mx_all = ot.gpu.to_gpu(mx_all) 
    dist = ot.gpu.to_gpu(dist.astype(float))
    lamb = ot.gpu.to_gpu(lamb)

    dist = dist.astype(float)
    
    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in x:
        ind = [y[1] for y in G.edges if y[0] == i]              

        W = ot.gpu.sinkhorn(mx_all[:,i].tolist(), mx_all[:,ind].tolist(), dist.tolist(), lamb)    
        if with_weights:
            Kt = np.append(Kt, ot.gpu.to_np(dist[i][ind] - W))
        else:
            Kt = np.append(Kt, 1. - W/ot.gpu.to_np(dist[i][ind]))
        
    return Kt
