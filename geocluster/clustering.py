#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import networkx as nx
from scipy.sparse import csr_matrix

'''
=============================================================================
Functions for clustering
=============================================================================
'''

def cluster_threshold(self, sample=20, perturb=0.02):
    #parameters
    #sample = 20     # how many samples to use for computing the VI
    #perturb = 0.02  # threshold k ~ Norm(0,perturb(kmax-kmin))

    Aold = self.A.toarray()
    K_tmp = nx.adjacency_matrix(self.G, weight='curvature').toarray()

    mink = np.min(K_tmp)
    maxk = np.max(K_tmp)
    labels = np.zeros([sample, Aold.shape[0] ])

    #set the first threshold to 0, others are random numbers around 0
    thres = np.append(0.0, np.random.normal(0, perturb*(maxk-mink), sample-1))

    nComms = np.zeros(sample)
    for k in range(sample):
        ind = np.where(K_tmp <= thres[k])     
        A = Aold.copy()
        A[ind[0], ind[1]] = 0 #remove edges with negative curvature.       
        nComms[k], labels[k] = sc.sparse.csgraph.connected_components(csr_matrix(A, dtype=int), directed=False, return_labels=True) 

    # compute the MI between the threshold=0 and other ones
    from sklearn.metrics.cluster import normalized_mutual_info_score

    mi = 0; k = 0 
    for i in range(sample):
            mi += normalized_mutual_info_score(list(labels[i]),list(self.labels_gt), average_method='arithmetic' )
            k+=1

    #return the mean number of communities, MI and label at threshold = 0 
    return np.mean(nComms), mi/k, labels[0]
