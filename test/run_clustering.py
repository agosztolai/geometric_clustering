import sys as sys
import os as os
import yaml as yaml
from geometric_clustering import Geometric_Clustering
from graph_generator import generate_graph

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#load parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)

#Set parameters
t_min = params['t_min'] #min Markov time
t_max = params['t_max'] #max Markov time
n_t = params['n_t']
cutoff = params['cutoff'] # truncate mx below cutoff*max(mx)
lamb = params['lamb'] # regularising parameter 
workers = 16 # numbers of cpus

#move to folder
os.chdir(graph_tpe)

# load graph 
G, pos  = generate_graph(tpe = graph_tpe, params = params)
         
# initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos=pos, t_min=t_min, t_max=t_max, n_t=n_t, \
                          log=True, cutoff=cutoff, workers=workers)

#load results
gc.load_curvature()

#cluster
gc.cluster_tpe = 'modularity' #'threshold'
gc.clustering()

#save it in a pickle
gc.save_clustering()

#plot the scan in time
gc.figsize = (10,8)
gc.node_labels= True
gc.plot_clustering()

#plot a graph snapshot per time
gc.video_clustering(n_plot = 100)