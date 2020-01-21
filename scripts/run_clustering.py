import sys as sys
import os 
import networkx as nx

from geocluster import geocluster

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#Load graph 
os.chdir(graph_tpe)
G = nx.read_gpickle(graph_tpe + "_0_.gpickle")

# initialise the code with parameters and graph 
gc = geocluster.GeoCluster(G)

#load results
gc.load_curvature()

#cluster 
#cluster_tpe: threshold, continuous_normalized (Markov stab), modularity_signed, linearized (Louvain))
#cluster_by: curvature, weight
gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
#gc.run_clustering(cluster_tpe='continuous_normalized', cluster_by='weight')

#save and plot
gc.save_clustering()
gc.plot_clustering()
gc.plot_graph_snapshots(folder='clustering_images', node_labels= False, cluster=True, node_size=10)
