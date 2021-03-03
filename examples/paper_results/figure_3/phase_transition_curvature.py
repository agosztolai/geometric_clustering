#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:47:08 2020

@author: adamgosztolai
"""

import numpy as np
import geocluster as gc
from graph_library import generate_GN
import pickle
import os
import networkx as nx

cases = 16
c = [5, 8, 10, 15, 20, 30] #average degree
n = 200
trials = 20
times = np.logspace(-2., 2., 25)
folder = '/data/AG/geocluster/phase_transition/'


for c_ in c:
    print('computing ' + str(c_))
    
    c_in = np.linspace(c_*0.5, c_*0.9, cases)
    c_out = c_ - c_in
    
    p_in = 2*c_in/n
    p_out = 2*c_out/n
        
    fname = folder + "phase_transition_curvature_final_k" + str(c_) + "_" + str(n) + ".pkl"
    if os.path.isfile(fname):
        print('results exists, continuing...')
        kappas = pickle.load(open(fname, "rb")) 
        count = len(kappas)*cases
    else:
        kappas = []
        count = 0

    kappas = []
    for j in range(trials):
        print('trial ' + str(j))
        seed = j
            
        kappa = []
        for i in range(cases):
            graph, _ = generate_GN({'l': 2, 'g': int(n/2), 'p_in': p_in[i], 'p_out': p_out[i]}, seed=int(count + i))
                
            # if not nx.is_connected(graph):
            #     kappa.append([np.NaN, np.NaN])
            #     continue
            
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc)
                
            kappatemp = gc.compute_curvatures(graph, 
                                              times, 
                                              use_spectral_gap=False,
                                              n_workers=1)
                
            kappa.append([graph, kappatemp])
    
        count += cases
        kappas.append(kappa)
                    
    pickle.dump(kappas, open(fname, "wb"))        