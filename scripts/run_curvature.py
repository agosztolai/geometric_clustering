import sys as sys
import os
import numpy as np

from geocluster import geocluster

import graph_library.graph_library as gl

#get the graph from terminal input 
graph_tpe = sys.argv[-1]     

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

#Load graph 
gg = gl.GraphGen(whichgraph=graph_tpe, 
                 paramsfile='./graph_params.yaml')
gg.generate()
         
#Initialise the code with parameters and graph 
os.chdir(graph_tpe)
T = np.logspace(gg.params['t_min'], 
                gg.params['t_max'], 
                gg.params['n_t'])
gc = geocluster.GeoCluster(gg.G, T=T, cutoff=1., workers=5, GPU=False, lamb=0.0, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.compute_OR_curvatures(with_weights=True)

#Save results for later analysis
gc.save_curvature()
gc.plot_edge_curvature()
gc.plot_graph_snapshots(folder='curvature_images', node_labels= False, cluster=False)
