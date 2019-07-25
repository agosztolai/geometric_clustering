import numpy as np
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx
import yaml as yaml
from graph_generator import generate_graph

#Set parameters
graph_tpe = 'LFR'
workers = 4 # numbers of cpus
folder = '/data/gosztolai/geocluster/LFR'
numGraphs = 25             # number of realisations
mu = np.linspace(0.,1.,6) #edge weights inside clusters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
GPU = 1

#run postprocess? 
postprocess = 0

#create a folder and move into it
if not os.path.isdir(folder):
    os.mkdir(folder)

os.chdir(folder)

batch_params = {}
if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and network realisations
    # =============================================================================
    for i in range(mu.shape[0]):
        for k in range(numGraphs):    
            batch_params['filename'] = 'graph_'+str(k)+'_mu_'+str(mu[i])
            print(batch_params['filename']) 
        
            # generate and save graph 
            batch_params['mu'] = mu[i]
            G, pos  = generate_graph(graph_tpe, params, batch_params)
            nx.write_gpickle(G, batch_params['filename']+".gpickle")
            
            # initialise the code with parameters and graph
            gc = Geometric_Clustering(G, t_min=params['t_min'], t_max=params['t_max'], n_t = params['n_t'], \
                          cutoff=params['cutoff'], workers=workers, filename = batch_params['filename'], GPU = GPU)
                 
            #First compute the geodesic distances
            gc.compute_distance_geodesic()

            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()

            #Save results for later analysis
            gc.save_curvature()
            
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(mu.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_mu_'+str(mu[i])
            print(filename) 
        
            # load graph 
            G = nx.read_gpickle(filename+".gpickle")
            
            # initialise the code with parameters and graph
            gc = Geometric_Clustering(G, t_min=params['t_min'], t_max=params['t_max'], n_t = params['n_t'], \
                          cutoff=params['cutoff'], workers=workers, filename = filename)
                 
            #load results        
            gc.load_curvature()

            #cluster
            gc.cluster_tpe = 'modularity'
            gc.clustering()

            #save it in a pickle
            gc.save_clustering()                