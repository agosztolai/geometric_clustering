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
graph_tpe = 'Girvan_Newman'
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
workers = 16 # numbers of cpus
folder = '/data/gosztolai/geocluster/' + graph_tpe
numGraphs = 100               # number of realisations
p_in = np.round(np.concatenate((np.linspace(0.15,0.34,20),np.linspace(0.36,0.42,4))),2)
p_out = (0.5-p_in)/3         # edge between clusters

#run postprocess? 
postprocess = int(sys.argv[-1])

#create a folder and move into it
if not os.path.isdir(folder):
    os.mkdir(folder)

os.chdir(folder)

if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and G-N realisations
    # =============================================================================
    for i in range(p_in.shape[0]):
        for k in range(numGraphs):   
            params['seed'] = k
            params['p_in'] = p_in[i]
            params['p_out'] = p_out[i]
            G = generate_graph(graph_tpe, params)
            G.graph['name'] = 'graph_'+str(k)+'_p_in_'+str(p_in[i])
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = Geometric_Clustering(G, T=T, workers=workers)
            
            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()
            save_curvature(gc)
   
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(p_in.shape[0]):
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_p_in_'+str(p_in[i])
            print(filename) 
      
            G = nx.read_gpickle(filename+".gpickle")
            gc = Geometric_Clustering(G)      
            load_curvature(gc)
            gc.run_lustering(cluster_tpe='modularity_signed', cluster_by='curvature')
            save_clustering(gc)