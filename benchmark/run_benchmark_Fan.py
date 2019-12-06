import networkx as nx
import sys as sys
sys.path.append('../utils')
import os as os
import yaml as yaml
from geometric_clustering import Geometric_Clustering
from graph_generator import generate_graph
from misc import save_curvature, save_clustering, load_curvature
import numpy as np

#Set parameters
graph_tpe = 'Fan'
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
workers = 40 # numbers of cpus
folder = '/disk2/Adam/geocluster/' + graph_tpe
#folder = '/data/gosztolai/geocluster/' + graph_tpe
numGraphs = 5               # number of realisations
w_in = np.round(np.concatenate((np.linspace(1.0,1.15,7),np.linspace(1.2,1.7,6))),3) #edge weights inside clusters

#run postprocess? 
postprocess = int(sys.argv[-1])

#create a folder and move into it
if not os.path.isdir(folder):
    os.mkdir(folder)

os.chdir(folder)

if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and network realisations
    # =============================================================================
    for i in range(w_in.shape[0]):
        for k in range(numGraphs):
            params['seed'] = k
            params['w_in'] = w_in[i]
            G = generate_graph(graph_tpe, params)
            G.graph['name'] = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = Geometric_Clustering(G, T=T, workers=workers)
                 
            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()

            #Save results for later analysis
            save_curvature(gc)
            
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(w_in.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
            print(filename) 
        
            G = nx.read_gpickle(filename+".gpickle")
            gc = Geometric_Clustering(G)        
            load_curvature(gc)
            gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
            save_clustering(gc)                