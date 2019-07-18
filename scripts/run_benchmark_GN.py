import numpy as np
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx
import yaml as yaml
from graph_generator import generate_graph

graph_tpe = 'Girvan_Newman'

#Set parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
l = params['l']         # clusters
g = params['g']         # vertices per group
workers = 16            # numbers of cpus
folder = '/data/gosztolai/geocluster/' + graph_tpe

#batch parameters
numGraphs = 100               # number of realisations
p_in = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, \
                 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.36, 0.38, 0.4, 0.42])
p_out = (0.5-p_in)/3         # edge between clusters

#run postprocess? 
postprocess = 1

#create a folder and move into it
if not os.path.isdir(folder):
    os.mkdir(folder)

os.chdir(folder)

batch_params = {}
if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and G-N realisations
    # =============================================================================
    for i in range(p_in.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_p_in_'+str(p_in[i])
            print(filename)      
            
            # generate and save graph 
            batch_params['p_in'] = p_in[i]
            batch_params['p_out'] = p_out[i]
            G, pos  = generate_graph(graph_tpe, params, batch_params)
            nx.write_gpickle(G, filename+".gpickle")
#            G = nx.planted_partition_graph(l, g, p_in[i], p_out[i], seed=1)
#            nx.write_gpickle(G, gc.filename+".gpickle") 
            
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
    for i in range(p_in.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_p_in_'+str(p_in[i])
            print(filename) 
        
            # load graph 
            G = nx.read_gpickle(filename+".gpickle")

            gc = Geometric_Clustering(G, t_min=params['t_min'], t_max=params['t_max'], n_t = params['n_t'], \
                          cutoff=params['cutoff'], workers=workers, filename = filename)
            
            #load results        
            gc.load_curvature()

            #cluster
            gc.cluster_tpe = 'modularity'
            gc.clustering()

            #save it in a pickle
            gc.save_clustering()