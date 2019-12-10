import sys as sys
import numpy as np
from geometric_clustering.geometric_clustering import Geometric_Clustering 
from geometric_clustering.utils import misc 
from graph_library import graph_generator as gg
import os

#get the graph from terminal input 
graph_tpe = sys.argv[-1]     

#Load graph 
G = gg(whichgraph=graph_tpe)
G.generate()
os.chdir(graph_tpe) 
         
#Initialise the code with parameters and graph 
T = np.logspace(G.params['t_min'], G.params['t_max'], G.params['n_t'])
gc = Geometric_Clustering(G.G, T=T, cutoff=1., workers=16, GPU=True, lamb=0.0, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.compute_OR_curvatures()

#Save results for later analysis
misc.save_curvature(gc)
misc.plot_graph_snapshots(gc, node_labels= False, cluster=False)