import numpy as np
from CurvCluster_parallel import CurvCluster_parallel
from inputGraphs import inputGraphs

# =============================================================================
# Parameters
# =============================================================================
T = np.logspace(0, 1, 20) # diffusion time scale 
cutoff = 1              # set mx(k) = 0 if mx(k) < (1-cutoff)* max_k( mx(k) )
lamb = 0                  # regularising parameter - set = 0 for exact 
                           # (the larger the more accurate, but higher cost, 
                           # and too large can blow up)                           
sample = 50                # how many samples to use for computing the VI
perturb = 0.1              # threshold k ~ Norm(0,perturb(kmax-kmin))
whichGraph = 3             # input graphs
workers = 1                # numbers of cpus
        
# load graph 
try:
    G
except NameError:
    (G,pos) = inputGraphs(whichGraph) 
         
# cluster
CurvCluster_parallel(G,pos,T,sample,cutoff,lamb,perturb)