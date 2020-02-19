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




def run_clustering(self, cluster_tpe='threshold', cluster_by='curvature'):
    """Clustering of curvature weigthed graphs"""
    
    self.cluster_tpe = cluster_tpe
    self.labels_gt = [int(self.G.nodes[i]['block']) for i in self.G.nodes if 'block' in self.G.nodes[0]]

    if cluster_tpe == 'threshold':
        
        nComms, MIs, labels = np.zeros(self.n_t), np.zeros(self.n_t), np.zeros([self.n_t, self.n])

        for t in tqdm(range(self.n_t)):
            # update edge curvatures                    
            for e, edge in enumerate(self.G.edges):
                self.G.edges[edge]['curvature'] = self.Kappa[e,t]                     

            nComms[t], MIs[t], labels[t] = cluster_threshold(self)
            
        self.clustering_results = {'Markov time' : self.T,
                'number_of_communities' : nComms,
                'community_id' : labels,
                'MI' : MIs}    

    else:
        import pygenstability.pygenstability as pgs

        #parameters
        stability = pgs.PyGenStability(self.G.copy(), cluster_tpe, louvain_runs=50)
        stability.all_mi = False #to compute MI between all Louvain runs
        stability.n_mi = 10  #if all_mi = False, number of top Louvain run to use for MI        
        stability.n_processes_louv = 10 #number of cpus 
        stability.n_processes_mi = 10 #number of cpus

        stabilities, nComms, MIs, labels = [], [], [], []
        for i in tqdm(range(self.n_t)):

            #set adjacency matrix
            if cluster_by == 'curvature':                 
                for e, edge in enumerate(self.G.edges):
                    stability.G.edges[edge]['curvature'] = self.Kappa[e,i] 
                stability.A = nx.adjacency_matrix(stability.G, weight='curvature')
                time = 1.
            elif cluster_by == 'weight' :   
                stability.A = nx.adjacency_matrix(stability.G, weight='weight')
                time = self.T[i]

            #run stability and collect results
            stability.run_single_stability(time = time ) 
            stabilities.append(stability.single_stability_result['stability'])
            nComms.append(stability.single_stability_result['number_of_comms'])
            MIs.append(stability.single_stability_result['MI'])
            labels.append(stability.single_stability_result['community_id'])

        ttprime = stability.compute_ttprime(labels, nComms, self.T[:np.shape(self.Kappa)[1]])

        #save the results
        self.clustering_results = {'Markov time' : self.T,
                'stability' : stabilities,
                'number_of_communities' : nComms,
                'community_id' : labels,
                'MI' : MIs, 
                'ttprime': ttprime}



