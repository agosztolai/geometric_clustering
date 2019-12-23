import sys as sys
import os
import numpy as np
from geocluster.geocluster import GeoCluster
from geocluster.utils import misc
from graph_library import graph_library as gl

#get the graph from terminal input 
graph_tpe = sys.argv[-1]     
gg = gl.GraphGen(whichgraph=graph_tpe, paramsfile='./graph_params.yaml')

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)

#Load graph 
gg.generate(similarity=gg.params['similarity'])
         
#Initialise the code with parameters and graph 
T = np.logspace(gg.params['t_min'], gg.params['t_max'], gg.params['n_t'])
gc = GeoCluster(gg.G, T=T, cutoff=1., workers=1, GPU=True, lamb=0.0, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.compute_OR_curvatures()

#Save results for later analysis
misc.save_curvature(gc)
misc.plot_graph_snapshots(gc, node_labels= False, cluster=False)
