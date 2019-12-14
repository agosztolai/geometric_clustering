import networkx as nx
import sys as sys
sys.path.append('../utils')
import os as os
import yaml as yaml
from geometric_clustering.geometric_clustering import Geometric_Clustering 
from geometric_clustering.utils import misc 
import numpy as np

#Set parameters
graph_tpe = 'LFR'
workers = 16 # numbers of cpus
folder = '/data/gosztolai/geocluster/' + graph_tpe
numGraphs = 20             # number of realisations
mu = [0.1,0.2,0.3,0.4,0.5] #edge weights inside clusters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]

#run postprocess? 
postprocess = 0

if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and network realisations
    # =============================================================================
    for _,i in enumerate(mu):
        os.chdir('/data/AG/geocluster/LFR/mu' + str(i))
        
        for k in range(numGraphs):    
            params['filename'] = 'LFR_'+str(k)+'_'
                   
            G = nx.read_gpickle(params['filename'] + ".gpickle")
            G.graph['name'] = 'graph_'+str(k)+'_mu_'+str()
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = Geometric_Clustering(G, T=T, workers=workers, cutoff=0.99, GPU=True, lamb=0.1)

            gc.compute_OR_curvatures()
            misc.save_curvature(gc)
            
        os.system('cd ..')
        
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