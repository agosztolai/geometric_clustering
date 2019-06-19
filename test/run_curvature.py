import sys as sys
import os as os
import yaml as yaml
from geometric_clustering import Geometric_Clustering
from graph_generator import generate_graph

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#Load parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)

#Set parameters
t_min = params['t_min'] #min Markov time
t_max = params['t_max'] #max Markov time
n_t = params['n_t'] #number of steps
cutoff = params['cutoff'] # truncate mx below cutoff*max(mx)
lamb = params['lamb'] # regularising parameter 
workers = 16 # numbers of cpus
GPU = 0 # use GPU?

#create a folder and move into it
if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)
        
#Load graph 
G, pos  = generate_graph(tpe = graph_tpe, params = params)
         
#Initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos=pos, t_min=t_min, t_max=t_max, n_t = n_t, \
                          log=True, cutoff=cutoff, workers=workers, GPU=GPU, lamb=lamb)

#First compute the geodesic distances
gc.compute_distance_geodesic()

#Compute the OR curvatures are all the times
gc.compute_OR_curvatures()

#Save results for later analysis
gc.save_curvature()