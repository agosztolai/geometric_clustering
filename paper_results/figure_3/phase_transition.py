#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:47:08 2020

@author: adamgosztolai
"""

import numpy as np
import scipy as sc
from graph_library import generate_GN
from tqdm import tqdm
import pickle
import networkx as nx
import os.path

from multiprocessing import Pool
from functools import partial

# compute all neighbourhood densities
def mx_comp(L, T, i):
    N = np.shape(L)[0]

    def delta(i, n):
        p0 = np.zeros(n)
        p0[i] = 1.
        return p0

    mx_all = [] 
    mx_tmp = delta(i, N) #set initial condition
    T = [0,] + list(T) #add time 0
    
    for i in range(len((T))-1): 
        #compute exponential by increments (faster than from 0)
        mx_tmp = sc.sparse.linalg.expm_multiply(-(T[i+1]-T[i])*L, mx_tmp)
        mx_all.append(mx_tmp)

    return np.array(mx_all)

#Compute diffusions
def fun(i, seed):    
    
    graph, _ = generate_GN({'l': 2, 'g': int(n/2), 'p_in': p_in[i], 'p_out': p_out[i]}, seed=int(seed + i))
    if not nx.is_connected(graph):
        return np.array([ np.NaN, np.NaN])
    
    degrees = np.array([graph.degree[i] for i in graph.nodes])
    L = nx.laplacian_matrix(graph).dot(sc.sparse.diags(1.0 / degrees))    
    
    #for source in range(int(n/2)):
    source = np.random.randint(int(n/2))
        
    m = mx_comp(L, times, source)
        
    mask_source = np.ones(n, dtype=bool)
    mask_target = np.ones(n, dtype=bool)
    
    mask_source[int(n/2):] = 0  
    mask_target[:int(n/2)] = 0
        
    mx = m[:,mask_source]
    mx[:, source] = 0
    peakin = np.mean(np.argmax(mx, axis=0))
        
    my = m[:,mask_target]
    peakout = np.mean(np.argmax(my, axis=0))

        
    return np.array([peakin, peakout])


cases = 16
c = [5, 8, 10, 15, 20, 30] #average degree
n = 400
trials = 200
times = np.logspace(-2., 2., 50)
folder = '/data/AG/geocluster/phase_transition/'

for c_ in c:
    print('computing', c_, 'in ', c) 
    c_in = np.linspace(c_*0.5, c_*0.9, cases)
    c_out = c_ - c_in
    
    p_in = 2*c_in/n
    p_out = 2*c_out/n

    fname = folder + "phase_transition_final_k" + str(c_) + "_" + str(n) + ".pkl"
    
    if os.path.isfile(fname):
        print('file exists, continuing...')
        ovs = pickle.load(open(fname, "rb")) 
        count = len(ovs)*cases
    else:
        ovs = []
        count = 0
        
    for j in tqdm(range(trials)):
        
        with Pool(processes = 16) as p_mx:
            ovs.append(list(p_mx.imap(partial(fun, seed=count), np.arange(cases)))) 
        
        count += cases
        
    pickle.dump(ovs, open(folder + "phase_transition_final_k" + str(c_) + "_" + str(n) + ".pkl", "wb"))        