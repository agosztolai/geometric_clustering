import networkx as nx
import os as os
from geocluster.geocluster import GeoCluster
from graph_library.graph_library import GraphGen
from geometric_clustering.utils import misc
import numpy as np

#Set parameters
graph_tpe = 'Fan'
workers = 16 # numbers of cpus
numGraphs = 100               # number of realisations
w_in = np.round(np.concatenate((np.linspace(1.0,1.15,7),np.linspace(1.2,1.7,6))),3)[::-1] #edge weights inside clusters

#run postprocess? 
postprocess = 1

if postprocess == 0:
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
        
        G = GraphGen(whichgraph='Fan', paramsfile='/home/gosztolai/Dropbox/github/geometric_clustering/benchmark/graph_params.yaml')
        G.outfolder = folder
        G.nsamples = numGraphs
        G.params['seed'] = i
        G.params['w_in'] = w_in[i]
        G.generate()
        
        for k in range(numGraphs):
            
            G.params['filename'] = 'Fan_'+str(k)+'_'
            G = nx.read_gpickle(G.params['filename'] + ".gpickle")
            G.graph['name'] = 'graph_'+str(k)+'_w_in_'+str(w_in[i])
            
            # initialise the code with parameters and graph
            T = np.logspace(G.params['t_min'], G.params['t_max'], G.params['n_t'])
            gc = GeoCluster(G, T=T, workers=workers,cutoff=1.,use_spectral_gap = False)
                 
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
            filename = 'Fan_'+str(k)+'_'
            print(filename) 
        
            G = nx.read_gpickle(filename+".gpickle")
            gc = GeoCluster(G)        
            misc.load_curvature(gc, filename='graph_' + str(k) + '_w_in_' + str(w_in[i]))
            gc.run_clustering(cluster_tpe='linearized', cluster_by='weight')
            misc.save_clustering(gc, filename='Fan_' + str(k))  
#            misc.plot_graph_snapshots(gc, node_labels= False, cluster=True)

        os.system('cd ..')             