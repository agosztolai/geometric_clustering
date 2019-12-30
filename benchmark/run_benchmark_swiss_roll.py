import networkx as nx
import os as os
from geocluster.geocluster import GeoCluster
from graph_library.graph_library import GraphGen
from geometric_clustering.utils import misc
import numpy as np

#Set parameters
workers = 16 # numbers of cpus
numGraphs = 30               # number of realisations
noise = np.linspace(.0,2.0,11)

#run postprocess? 
postprocess = 0

if postprocess == 0:
    # =============================================================================
    # Main loop: repeat for all parameters and G-N realisations
    # =============================================================================
    for i in range(noise.shape[0]):
        folder = '/data/AG/geocluster/swiss-roll/noise_' + str(noise[i])
        #create a folder and move into it
        if not os.path.isdir(folder):
            os.mkdir(folder)
        os.chdir(folder)
        
        print(folder)
        
        G = GraphGen(whichgraph='swiss-roll', paramsfile='/home/gosztolai/Dropbox/github/geometric_clustering/benchmark/graph_params.yaml')
        G.outfolder = folder
        G.nsamples = numGraphs
        G.params['seed'] = i
        G.params['noise'] = noise[i]
        G.generate(similarity = 'knn')
        
        print(G.params['t_min'])
        for k in range(numGraphs):   
            
            G.params['filename'] = 'swiss-roll_'+str(k)+'_'
            graph = nx.read_gpickle(G.params['filename'] + ".gpickle")
            graph.graph['name'] = 'swiss-roll_'+str(k)
            
            # initialise the code with parameters and graph
            T = np.logspace(G.params['t_min'], G.params['t_max'], G.params['n_t'])
            gc = GeoCluster(G, T=T, workers=workers, cutoff=1., GPU=True, lamb=0.1, use_spectral_gap = False)
                 
            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()

            #Save results for later analysis
            misc.save_curvature(gc)
            
        os.system('cd ..')    
   
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(noise.shape[0]):
        folder = '/data/AG/geocluster/swiss-roll/noise_' + str(noise[i])
        os.chdir(folder)
        
        for k in range(numGraphs):
            filename = 'swiss-roll_'+str(k)+'_'
            print(filename + str(noise[i])) 
        
            G = nx.read_gpickle(filename+".gpickle")
            gc = GeoCluster(G)        
            misc.load_curvature(gc, filename='swiss-roll_' + str(k))
            gc.run_embedding()
            misc.save_clustering(gc, filename='swiss-roll_' + str(k))  
            misc.plot_graph_snapshots(gc, node_labels= False, cluster=True)
            
        os.system('cd ..') 