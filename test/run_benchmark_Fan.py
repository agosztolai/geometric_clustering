import numpy as np
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx

#Set parameters
t_min = 0. #min Markov time
t_max = 1. #max Markov time
n_t = 20
cutoff = 0.99 # truncate mx below cutoff*max(mx)
workers = 44 # numbers of cpus

# =============================================================================
# Parameters for Fan benchmark graph
# =============================================================================
numGraphs = 50               # number of realisations
l = 4                        # clusters
g = 32                       # vertices per group
w_in = np.linspace(1.5,1.9,9) #<0.5 - edge inside clusters
p_out = 1/8         # edge between clusters
p_in = 1/8

folder = '/disk2/Adam/geocluster/Fan'

#create a folder and move into it
if not os.path.isdir(folder):
    os.mkdir(folder)

os.chdir(folder)

# =============================================================================
# Main loop: repeat for all parameters and network realisations
# =============================================================================
for i in range(w_in.shape[0]):
    for k in range(numGraphs):
        filename = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
        
        # load and save graph 
        G = nx.planted_partition_graph(l, g, p_in, p_out, seed=2)
        for edge in G.edges:
            if G.node[edge[0]]['block'] == G.node[edge[1]]['block']:
                G.edges[edge]['weight'] = w_in[i]
            else:
                G.edges[edge]['weight'] = 2 - w_in[i]
                    
        nx.write_gpickle(G, filename+".gpickle")
                 
        # initialise the code with parameters and graph 
        gc = Geometric_Clustering(G, t_min=t_min, t_max=t_max, n_t=n_t,\
                                  cutoff=cutoff, workers=workers, filename=filename)

        #First compute the geodesic distances
        gc.compute_distance_geodesic()

        #Compute the OR curvatures are all the times
        gc.compute_OR_curvatures()

        #Save results for later analysis
        gc.save_curvature()