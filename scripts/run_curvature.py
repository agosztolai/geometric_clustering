import sys as sys
import os
import numpy as np
import yaml
from geocluster import GeoCluster
from graph_library import generate

#get the graph from terminal input 
whichgraph = sys.argv[-1]     

#load parameters
paramsfile='graph_params.yaml'
params = yaml.load(open(paramsfile,'rb'), Loader=yaml.FullLoader)[whichgraph]

if not os.path.isdir(whichgraph):
    os.mkdir(whichgraph)

os.chdir(whichgraph)

#Load graph 
G = generate(whichgraph=whichgraph, params=params)
         
#Initialise the code with parameters and graph 
T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
gc = GeoCluster(G, T=T, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.compute_OR_curvatures(with_weights=False, workers=10, GPU=False, cutoff=1.-1e-5, lamb=0.0)

#Save results for later analysis
gc.save_curvature()
gc.plot_edge_curvature()
gc.plot_graph_snapshots(folder='curvature_images', node_labels=False, cluster=False)
