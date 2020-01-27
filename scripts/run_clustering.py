import sys as sys
import os 
import yaml
import numpy as np
from geocluster import GeoCluster
from graph_library import generate

#get the graph from terminal input 
whichgraph = sys.argv[-1]     

#load parameters
paramsfile='graph_params.yaml'
params = yaml.load(open(paramsfile,'rb'), Loader=yaml.FullLoader)[whichgraph]

os.chdir(whichgraph)

#Load graph 
G = generate(whichgraph=whichgraph, params=params)
         
#Initialise the code with parameters and graph 
T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
gc = GeoCluster(G, T=T, cutoff=1.-1e-5, workers=1, GPU=False, lamb=0.0, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.load_curvature()

#cluster 
#cluster_tpe: threshold, continuous_normalized (Markov stab), modularity_signed, linearized (Louvain))
#cluster_by: curvature, weight
gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
#gc.run_clustering(cluster_tpe='continuous_normalized', cluster_by='weight')

#save and plot
gc.save_clustering()
gc.plot_clustering()
gc.plot_graph_snapshots(folder='clustering_images', node_labels=False, cluster=True, node_size=50)
