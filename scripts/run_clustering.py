import sys as sys
sys.path.append('../utils')
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx
from misc import load_curvature, save_clustering, plot_clustering, plot_graph_snapshots

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#load graph 
os.chdir(graph_tpe)
G = nx.read_gpickle(graph_tpe + ".gpickle")
         
# initialise the code with parameters and graph 
gc = Geometric_Clustering(G)

#load results
load_curvature(gc)

#cluster 
#cluster_tpe: threshold, continuous_normalized (Markov stab), modularity_signed, linearized (Louvain))
#cluster_by: curvature, weight
gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')

#save and plot
save_clustering(gc)
plot_clustering(gc)
plot_graph_snapshots(gc, node_labels= False, cluster=True)
