import sys as sys
import os as os
import yaml as yaml
from geometric_clustering import Geometric_Clustering
from graph_generator import generate_graph
import networkx as nx

#get the graph from terminal input 
graph_tpe = sys.argv[-1]
#graph_tpe = 'LFR'

#Load parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)
workers = 2 # numbers of cpus
GPU = 0 # use GPU?

#create a folder and move into it
if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)
        
#Load graph 
G, pos  = generate_graph(tpe = graph_tpe, params = params)
         
#Initialise the code with parameters and graph 
print("Create the geometric clustering object")
gc = Geometric_Clustering(G, pos=pos, t_min=params['t_min'], t_max=params['t_max'], n_t = params['n_t'], \
                          cutoff=params['cutoff'], workers=workers, GPU=GPU, lamb=params['lamb'])

#First compute the geodesic distances
print("Compute geodesic distances")
gc.compute_distance_geodesic()

#Compute the OR curvatures are all the times
print("Compute OR curvatures")
gc.compute_OR_curvatures()

#Save results for later analysis
gc.save_curvature()

nx.write_gpickle(G, graph_tpe + ".gpickle")

#plotting 
gc.plot_curvatures()
gc.video_curvature()

