import numpy as np
import sys as sys
import os as os
import yaml as yaml

from geometric_clustering import Geometric_Clustering

from graph_generator import generate_graph


#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#load parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)


# =============================================================================
# Set parameters
# =============================================================================

# diffusion time scale 
t_min = params['t_min']
t_max = params['t_max']
n_t = params['n_t']

# set mx(k) = 0 if mx(k) < (1-cutoff)* max_k( mx(k) )
cutoff = params['cutoff']

# regularising parameter - set = 0 for exact 
# (the larger the more accurate, but higher cost, 
# and too large can blow up)                           
lamb = params['lamb']


workers = 16               # numbers of cpus
GPU = 1

#move to folder
os.chdir(graph_tpe)

# load graph 
G, pos  = generate_graph(tpe = graph_tpe, params = params)
         
# initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos = pos, t_min = t_min, t_max = t_max, n_t = n_t, log = True, cutoff = cutoff, lamb = lamb)

#load results
gc.load_curvature()

gc.cluster_tpe = 'threshold'

#apply the clustering
gc.clustering()

#save it in a pickle
gc.save_clustering()

#plot the scan in time
gc.figsize = (5,4)
gc.plot_clustering()

#plot a graph snapshot per time
gc.video_clustering(n_plot = 20)


