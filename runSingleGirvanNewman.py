import numpy as np
from CurvCluster_parallel import CurvCluster_parallel
import networkx as nx

# =============================================================================
# General parameters
# =============================================================================
T = np.logspace(-2.0, 2.0, 50) # diffusion time scale 
cutoff = 1.0              # set mx(k) = 0 if mx(k) < (1-cutoff)* max_k( mx(k) )
lamb = 20                    # regularising parameter - set = 0 for exact                          
sample = 100                 # how many samples to use for computing the VI
perturb = 0.1                # threshold k ~ Norm(0,perturb(kmax-kmin))
workers = 2                 # numbers of cpus

louvain = True
vis = 1

# =============================================================================
# Parameters for Girvan-Newman graph
# =============================================================================
numGraphs = 50              # number of realisations
l = 3                        # clusters
g = 10                       # vertices per group
p_in = 0.8           #<0.5 - edge inside clusters
p_out = 0.1#(0.5-p_in)/3         # edge between clusters

# =============================================================================
# Main loop: repeat for all parameters and G-N realisations
# =============================================================================
# load graph 
G = nx.planted_partition_graph(l, g, p_in, p_out, seed=0)
pos = nx.spring_layout(G)
         
# cluster
CurvCluster_parallel(G,pos,T,sample,cutoff,lamb,perturb,workers,louvain=louvain, vis = vis)
