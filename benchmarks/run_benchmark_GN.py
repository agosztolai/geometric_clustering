import numpy as np
import os as os
from geometric_clustering import Geometric_Clustering
import networkx as nx
import yaml as yaml
from graph_generator import generate_graph

#Set parameters
graph_tpe = 'Girvan_Newman'
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
workers = 16 # numbers of cpus
#folder = '/data/gosztolai/geocluster/' + graph_tpe
folder = '/disk2/Adam/geocluster/' + graph_tpe
numGraphs = 100               # number of realisations
p_in = np.round(np.concatenate((np.linspace(0.15,0.34,20),np.linspace(0.36,0.42,4))),2)
p_out = (0.5-p_in)/3         # edge between clusters

#run postprocess? 
postprocess = 0

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
            params['seed'] = k
            
            # generate and save graph 
            batch_params['p_in'] = p_in[i]
            batch_params['p_out'] = p_out[i]
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