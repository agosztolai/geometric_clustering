import numpy as np
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx

#Set parameters
t_min = -0.5 #min Markov time
t_max = 0.5 #max Markov time
n_t = 20
cutoff = 0.99 # truncate mx below cutoff*max(mx)
workers = 44 # numbers of cpus

# =============================================================================
# Parameters for Girvan-Newman graph
# =============================================================================
numGraphs = 50               # number of realisations
l = 4                        # clusters
g = 32                       # vertices per group
p_in = np.linspace(0.34,0.42,5) #<0.5 - edge inside clusters
p_out = (0.5-p_in)/3         # edge between clusters

folder = '/disk2/Adam/geocluster/Girvan_Newman'

#create a folder and move into it
if not os.path.isdir(folder):
    os.mkdir(folder)

os.chdir(folder)

# =============================================================================
# Main loop: repeat for all parameters and G-N realisations
# =============================================================================
for i in range(p_in.shape[0]):
    for k in range(numGraphs):
        filename = 'graph_'+str(k)+'_p_in_'+str(p_in[i])
        
        # load and save graph 
        G = nx.planted_partition_graph(l, g, p_in[i], p_out[i], seed=2)
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