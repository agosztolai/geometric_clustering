import sys as sys
#sys.path.append('../utils')
from geometric_clustering import Geometric_Clustering
from graph_library import graph_generator as gg
from misc import save_curvature, plot_graph_snapshots
import numpy as np

#get the graph from terminal input 
graph_tpe = sys.argv[-1]
      
#Load graph 
G = gg.generate()
G.whichgraph = graph_tpe
G.outfolder = '../results/'
G.paramsfile = '../utils/graph_params.yaml'
         
#Initialise the code with parameters and graph 
T = np.logspace(G.params['t_min'], G.params['t_max'], G.params['n_t'])
gc = Geometric_Clustering(G, T=T, cutoff=0.95, workers=16, GPU=True, lamb=0.5, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.compute_OR_curvatures()

#Save results for later analysis
save_curvature(gc)
plot_graph_snapshots(gc, node_labels= False, cluster=False)