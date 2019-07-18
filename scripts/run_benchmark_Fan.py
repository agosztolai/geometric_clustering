import numpy as np
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx
import yaml as yaml
from graph_generator import generate_graph

#Set parameters
graph_tpe = 'Fan'
workers = 16 # numbers of cpus
folder = '/data/gosztolai/geocluster/Fan'
numGraphs = 50               # number of realisations
w_in = np.linspace(1.80,1.95,7) #edge weights inside clusters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]

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
    for i in range(w_in.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
            print(filename) 
        
            # generate and save graph 
            batch_params['w_in'] = w_in[i]
            G, pos  = generate_graph(graph_tpe, params, batch_params)
            nx.write_gpickle(G, filename+".gpickle")
            
            # initialise the code with parameters and graph
            gc = Geometric_Clustering(G, t_min=params['t_min'], t_max=params['t_max'], n_t = params['n_t'], \
                          cutoff=params['cutoff'], workers=workers, filename = filename)
                 
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
    for i in range(w_in.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
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