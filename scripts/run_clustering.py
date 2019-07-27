import sys as sys
import os as os
import yaml as yaml
from geometric_clustering import Geometric_Clustering
import networkx as nx

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#load parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)

#move to folder
os.chdir(graph_tpe)

# load graph 
G = nx.read_gpickle(graph_tpe + ".gpickle")
pos = []
         
# initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos=pos, t_min=params['t_min'], t_max=params['t_max'], n_t=params['n_t'], \
                          cutoff=params['cutoff'], workers=16)

#load results
gc.load_curvature()

#cluster
gc.cluster_tpe = 'modularity' #'threshold'
gc.clustering()

#save it in a pickle
gc.save_clustering()

#plot the scan in time
gc.plot_clustering()

#plot a graph snapshot per time
gc.video_clustering(n_plot = 100, node_labels= False)
