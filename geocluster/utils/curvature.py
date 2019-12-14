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
def mx_comp(L, T, cutoff, i):
    N = np.shape(L)[0]

    def delta(i, n):
        p0 = np.zeros(n)
        p0[i] = 1.
        return p0

    mx_all = []
    Nx_all = []

    mx_tmp = delta(i, N) #set initial condition
    T = [0,] + list(T) #add time 0
    for i in range(len((T))-1): 
        #compute exponential by increments (faster than from 0)
        mx_tmp = sc.sparse.linalg.expm_multiply(-(T[i+1]-T[i])*L, mx_tmp)

        Nx = np.argwhere(mx_tmp >= (1-cutoff)*np.max(mx_tmp))
        mx_all.append(sc.sparse.lil_matrix(mx_tmp[Nx]/np.sum(mx_tmp[Nx])))
        Nx_all.append(Nx)

    return mx_all, Nx_all


# compute curvature for an edge ij
def K_ij(mx_all, dist, lamb, e):
    i = e[0]
    j = e[1]

    nt = len(mx_all[0][0])
    K = np.zeros(nt)
    for it in range(nt):

        Nx = np.array(mx_all[i][1][it]).flatten()
        Ny = np.array(mx_all[j][1][it]).flatten()
        mx = mx_all[i][0][it].toarray().flatten()
        my = mx_all[j][0][it].toarray().flatten()

        dNxNy = dist[Nx,:][:,Ny].copy(order='C')

        if lamb != 0: #entropy regularized OT
            W = ot.sinkhorn2(mx, my, dNxNy, lamb)
        elif lamb == 0: #classical sparse OT
            W = ot.emd2(mx, my, dNxNy) 
            
        #K[it] = 1. - W/dist[i, j]  
        K[it] = dist[i, j]  - W
         
    return K


def K_all(mx_all, dist, lamb, G):     
       
    dist = dist.astype(float)
    
    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in tqdm(x):
        ind = [y[1] for y in G.edges if y[0] == i]              

        W = ot.sinkhorn(mx_all[:,i].tolist(), mx_all[:,ind].tolist(), dist.tolist(), lamb)    
        #Kt = np.append(Kt, 1. - W/dist[i][ind])
        Kt = np.append(Kt, dist[i][ind] - W)
        
    return Kt


def K_all_gpu(mx_all, dist, lamb, G):   
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
        Kt = np.append(Kt, 1. - W/ot.gpu.to_np(dist[i][ind]))
        
    return Kt