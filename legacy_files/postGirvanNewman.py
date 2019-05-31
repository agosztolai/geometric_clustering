import numpy as np
from funs import loaddata
#import networkx as nx
from sklearn import metrics
import matplotlib.pyplot as plt

# =============================================================================
# Parameters for Girvan-Newman graph
# =============================================================================
numGraphs = 50              # number of realisations
p_in = np.linspace(0.26,0.48,12) #<0.5 - edge inside clusters
p_out = (0.5-p_in)/3         # edge between clusters

# =============================================================================
# Compute performance
# =============================================================================
nMI = np.zeros(p_in.shape[0])   
mean_nMI = np.zeros(p_in.shape[0])     
std_nMI = np.zeros(p_in.shape[0])     
for i in range(p_in.shape[0]):
    nMItemp = np.zeros(numGraphs)
    for k in range(numGraphs):
        
        # load data
        filename = 'graph_'+str(k)+'_p_in_'+str(p_in[i])  
        (T,nComms,vi,data) = loaddata(filename)
        
        # take partition with lowest vi and compute the norm MI
        ind = np.argmin(vi[6:])
        nMItemp[k] = metrics.normalized_mutual_info_score(data[:,0],data[:,7+ind])
        
    mean_nMI[i] = np.mean(nMItemp)
    std_nMI[i] = np.std(nMItemp)
    
# =============================================================================
# Plot    
# =============================================================================
plt.errorbar(p_in,mean_nMI,std_nMI,marker='o')
plt.xlabel("Proportion of outlinks z_out/k")
plt.ylabel("Normalised mutual information")