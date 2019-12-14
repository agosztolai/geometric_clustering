import sys as sys
import numpy as np
import geometric_clustering as gc
from geometric_clustering.utils import misc 
import graph_library as gl 

#get the graph from terminal input 
graph_tpe = sys.argv[-1]     

#Load graph 
gg = gl.graph_generator(whichgraph=graph_tpe, paramsfile='./graph_params.yaml')
gg.generate(similarity=gg.params['similarity'])
         
#Initialise the code with parameters and graph 
T = np.logspace(gg.params['t_min'], gg.params['t_max'], gg.params['n_t'])
gc = gc(gg.G, T=T, cutoff=1., workers=1, GPU=True, lamb=0.0, laplacian_tpe='normalized')

#Compute the OR curvatures
gc.compute_OR_curvatures()

#Save results for later analysis
misc.save_curvature(gc)
misc.plot_graph_snapshots(gc, node_labels= False, cluster=False)
