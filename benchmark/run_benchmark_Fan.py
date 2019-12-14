import networkx as nx
import sys as sys
sys.path.append('../utils')
import os as os
from geometric_clustering.geometric_clustering import Geometric_Clustering
from graph_library import graph_generator as gg
from geometric_clustering.utils import misc
import numpy as np
import yaml

#Set parameters
graph_tpe = 'Fan'
workers = 16 # numbers of cpus
folder = '/disk2/Adam/geocluster/' + graph_tpe
#folder = '/data/gosztolai/geocluster/' + graph_tpe
numGraphs = 100               # number of realisations
w_in = np.round(np.concatenate((np.linspace(1.0,1.15,7),np.linspace(1.2,1.7,6))),3)[::-1] #edge weights inside clusters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]

#run postprocess? 
postprocess = 0

if postprocess != 1:
    # =============================================================================
    # Main loop: repeat for all parameters and network realisations
    # =============================================================================
    for i in range(w_in.shape[0]):
        folder = '/data/AG/geocluster/Fan/w_in_' + str(w_in[i])
        #create a folder and move into it
        if not os.path.isdir(folder):
            os.mkdir(folder)
        os.chdir(folder)
        
        print(folder)
        
        G = gg(whichgraph='Fan')
        G.outfolder = folder
        G.nsamples = numGraphs
        G.params['seed'] = i
        G.params['w_in'] = w_in[i]
        G.generate()
        
        for k in range(numGraphs):
            
            params['filename'] = 'Fan_'+str(k)+'_'
            G = nx.read_gpickle(params['filename'] + ".gpickle")
            G.graph['name'] = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = Geometric_Clustering(G, T=T, workers=workers,cutoff=1.)
                 
            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()

            #Save results for later analysis
            misc.save_curvature(gc)
            
        os.system('cd ..')  
        
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(w_in.shape[0]):
        folder = '/data/AG/geocluster/Fan/w_in_' + str(w_in[i])
        os.chdir(folder)
        
        for k in range(numGraphs):
            filename = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
            print(filename) 
        
            G = nx.read_gpickle(filename+".gpickle")
            gc = Geometric_Clustering(G)        
            misc.load_curvature(gc)
            gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
            misc.save_clustering(gc)   

        os.system('cd ..')             