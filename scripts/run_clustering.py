import sys as sys
import os 
import geocluster
from geocluster.utils import misc 
import networkx as nx

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#load graph 
os.chdir(graph_tpe)
G = nx.read_gpickle(graph_tpe + ".gpickle")
         
# initialise the code with parameters and graph 
gc = geocluster(G)

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
