import numpy as np
from CurvCluster_parallel import CurvCluster_parallel
import networkx as nx
import os        

# =============================================================================
# General parameters
# =============================================================================
T = np.logspace(-0.3, 0, 15) # diffusion time scale 
cutoff = 0.95                # set mx(k) = 0 if mx(k) < (1-cutoff)* max_k( mx(k) )
lamb = 1                    # regularising parameter - set = 0 for exact                          
sample = 100                 # how many samples to use for computing the VI
perturb = 0.1                # threshold k ~ Norm(0,perturb(kmax-kmin))
workers = 1                 # numbers of cpus
GPU = 0

# =============================================================================
# Parameters for Girvan-Newman graph
# =============================================================================
numGraphs = 50               # number of realisations
l = 4                        # clusters
g = 32                       # vertices per group
p_in = np.linspace(0.34,0.42,5) #<0.5 - edge inside clusters
p_out = (0.5-p_in)/3         # edge between clusters

# =============================================================================
# Main loop: repeat for all parameters and G-N realisations
# =============================================================================
for i in range(p_in.shape[0]):
    for k in range(numGraphs):
        filename = 'graph_'+str(k+100)+'_p_in_'+str(p_in[i])
        
        # load and save graph 
        G = nx.planted_partition_graph(l, g, p_in[i], p_out[i], seed=2)
        nx.write_adjlist(G,os.getcwd()+'/data/graph_'+str(i)+'.adjlist')
        pos = nx.spring_layout(G)
         
        # cluster
        CurvCluster_parallel(G,pos,T,sample,cutoff,lamb,perturb,workers,GPU,filename)