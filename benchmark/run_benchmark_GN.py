import networkx as nx
import os as os
from geocluster.geocluster import GeoCluster
from graph_library.graph_library import GraphGen
from geometric_clustering.utils import misc
import numpy as np

#Set parameters
workers = 16 # numbers of cpus
numGraphs = 100               # number of realisations
p_in = np.round(np.concatenate((np.linspace(0.15,0.34,20),np.linspace(0.36,0.42,4))),2)
p_out = (0.5-p_in)/3         # edge between clusters

#run postprocess? 
postprocess = 0

if postprocess == 0:
    # =============================================================================
    # Main loop: repeat for all parameters and G-N realisations
    # =============================================================================
    for i in range(p_in.shape[0]):
        folder = '/data/AG/geocluster/GN/pin_' + str(p_in[i])
        #create a folder and move into it
        if not os.path.isdir(folder):
            os.mkdir(folder)
        os.chdir(folder)
        
        print(folder)
        
        G = GraphGen(whichgraph='GN', paramsfile='/home/gosztolai/Dropbox/github/geometric_clustering/benchmark/graph_params.yaml')
        G.outfolder = folder
        G.nsamples = numGraphs
        G.params['seed'] = i
        G.params['p_in'] = p_in[i]
        G.params['p_out'] = p_out[i]
        G.generate()
        
        print(G.params['t_min'])
        for k in range(numGraphs):   
            
            G.params['filename'] = 'GN_'+str(k)+'_'
            graph = nx.read_gpickle(G.params['filename'] + ".gpickle")
            graph.graph['name'] = 'GN_'+str(k)
            
            # initialise the code with parameters and graph
            T = np.logspace(G.params['t_min'], G.params['t_max'], G.params['n_t'])
            gc = GeoCluster(graph, T=T, workers=workers, cutoff=1., use_spectral_gap = False)
                 
            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()

            #Save results for later analysis
            misc.save_curvature(gc)
            
        os.system('cd ..')    
   
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(p_in.shape[0]):
        folder = '/data/AG/geocluster/GN/pin_' + str(p_in[i])
        os.chdir(folder)
        
        for k in range(numGraphs):
            filename = 'GN_'+str(k)+'_'
            print(filename + str(p_in[i])) 
        
            G = nx.read_gpickle(filename+".gpickle")
            gc = GeoCluster(G)        
            misc.load_curvature(gc, filename='GN_' + str(k))
            gc.run_clustering(cluster_tpe='threshold', cluster_by='curvature')
            misc.save_clustering(gc, filename='GN_' + str(k))  
            misc.plot_graph_snapshots(gc, node_labels= False, cluster=True)
            
        os.system('cd ..') 