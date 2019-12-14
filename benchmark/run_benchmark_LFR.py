import networkx as nx
import os as os
import yaml as yaml
import geocluster
from utils import misc 
import numpy as np

#Set parameters
graph_tpe = 'LFR'
workers = 16 # numbers of cpus
folder = '/data/gosztolai/geocluster/' + graph_tpe
numGraphs = 30             # number of realisations
mu = [0.1,0.2,0.3,0.4,0.5] #edge weights inside clusters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]

#run postprocess? 
postprocess = 0

if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and network realisations
    # =============================================================================
    for _,i in enumerate(mu):
        os.chdir('/data/AG/geocluster/LFR/tau_1_2_tau_2_2/mu' + str(i))
        print(i)
        
        for k in range(numGraphs):    
            params['filename'] = 'LFR_'+str(k)+'_'
                   
            G = nx.read_gpickle(params['filename'] + ".gpickle")
            G.graph['name'] = 'graph_'+str(k)+'_mu_'+str(i)
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = geocluster(G, T=T, workers=workers, cutoff=1., GPU=True, lamb=0.1)

            gc.compute_OR_curvatures()
            misc.save_curvature(gc)
            
        os.system('cd ..')
        
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for _,i in enumerate(mu):
        os.chdir('/data/AG/geocluster/LFR/mu' + str(i))
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_mu_'
            print(filename) 

            G = nx.read_gpickle(filename+".gpickle")
            gc = Geometric_Clustering(G)      
            misc.load_curvature(gc)
            gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
            misc.save_clustering(gc)    
            misc.plot_clustering(gc)  
            misc.plot_graph_snapshots(gc, node_labels= False, cluster=True)
            
        os.system('cd ..')     