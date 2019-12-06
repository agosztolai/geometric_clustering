import sys as sys
import os 

from geometric_clustering import geometric_clustering 
from geometric_clustering.utils import misc 
import graph_library as gl

import networkx as nx

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

##load graph 
#os.chdir(graph_tpe)
#G = nx.read_gpickle(graph_tpe + ".gpickle")

#Load graph 
gg = gl.graph_generator()
gg.generate()
gg.whichgraph = graph_tpe
gg.outfolder = '../results/'
gg.paramsfile = '../utils/graph_params.yaml'
         
# initialise the code with parameters and graph 
gc = geometric_clustering.Geometric_Clustering(gg.G)

#load results
misc.load_curvature(gc)

#cluster 
#cluster_tpe: threshold, continuous_normalized (Markov stab), modularity_signed, linearized (Louvain))
#cluster_by: curvature, weight
gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
#gc.run_clustering(cluster_tpe='continuous_normalized', cluster_by='weight')

#save and plot
misc.save_clustering(gc)
misc.plot_clustering(gc)
misc.plot_graph_snapshots(gc, node_labels= False, cluster=True)
