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
graph_tpe = 'LFR'
workers = 16 # numbers of cpus
folder = '/data/gosztolai/geocluster/' + graph_tpe
numGraphs = 20             # number of realisations
mu = np.linspace(0.1,0.9,5) #edge weights inside clusters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]

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
    for i in range(mu.shape[0]):
        for k in range(numGraphs):    
            params['filename'] = 'graph_'+str(k)+'_mu_'+str(mu[i])
            params['mu'] = 0.2#mu[i]
                   
            G = generate_graph(graph_tpe, params)
            G.graph['name'] = 'graph_'+str(k)+'_mu_'+str(mu[i])
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = Geometric_Clustering(G, T=T, workers=workers)
            gc = Geometric_Clustering(G, cutoff=0.95, workers=workers, GPU=True, lamb=0.1)

            gc.compute_OR_curvatures()
            save_curvature(gc)
            
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(mu.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_mu_'+str(mu[i])
            print(filename) 

            G = nx.read_gpickle(filename+".gpickle")
            gc = Geometric_Clustering(G)      
            load_curvature(gc)
            gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
            save_clustering(gc)                