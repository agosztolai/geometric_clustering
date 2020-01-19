import networkx as nx
import os as os
from geocluster import GeoCluster
from graph_library import generate_Fan
import numpy as np
import yaml

#Set parameters
workers = 16 # numbers of cpus
numGraphs = 100               # number of realisations
w_in = np.round(np.concatenate((np.linspace(1.0,1.15,7),np.linspace(1.2,1.7,6))),3)[::-1] #edge weights inside clusters
paramsfile='/home/gosztolai/Dropbox/github/geometric_clustering/benchmark/graph_params.yaml'
params = yaml.load(open(paramsfile,'rb'), Loader=yaml.FullLoader)['Fan']
        
#run postprocess? 
postprocess = 0

if postprocess == 0:
    # =============================================================================
    # Main loop: repeat for all parameters and network realisations
    # =============================================================================
    for i in range(w_in.shape[0]):
        
        #create a folder and move into it
        folder = '/data/AG/geocluster/Fan/win_' + str(w_in[i])
        if not os.path.isdir(folder):
            os.mkdir(folder)
        os.chdir(folder)
        
        print(folder)
        
        #Generate graphs
        if not os.path.isfile('Fan_0_.gpickle'):
#            G.outfolder = folder
#            G.nsamples = numGraphs
#            G.params['seed'] = i #change seed every time
#            G.params['w_in'] = w_in[i]
#            generate(whichgraph='Fan')
            for j in range(numGraphs):
                G, _, _ = generate_Fan(params = {'w_in': w_in[i], 'l': 4, 'g': 32, 'p_in': 0.125, 'p_out': 0.125, 'seed': j})
                nx.write_gpickle(G, "Fan_" + str(j) + "_.gpickle")
            
        for k in range(numGraphs):
            
            if os.path.isfile('Fan_'+str(k)+'_curvature.pkl'):
                continue
            
            params['filename'] = 'Fan_'+str(k)+'_'
            graph = nx.read_gpickle(G.params['filename'] + ".gpickle")
            graph.graph['name'] = 'Fan_'+str(k)
            
            # initialise the code with parameters and graph
            T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
            gc = GeoCluster(graph, T=T, workers=workers, cutoff=1., use_spectral_gap = False)
                 
            #Compute the OR curvatures are all the times
            gc.compute_OR_curvatures()

            #Save results for later analysis
            gc.save_curvature(gc)
            
        os.system('cd ..')  
        
if postprocess == 1:     
    # =============================================================================
    # Postprocess
    # ============================================================================= 
    for i in range(w_in.shape[0]):
        folder = '/data/AG/geocluster/Fan/win_' + str(w_in[i])
        os.chdir(folder)
        
        for k in range(numGraphs):
            filename = 'Fan_'+str(k)+'_'
            print(filename) 
        
            G = nx.read_gpickle(filename+".gpickle")
            gc = GeoCluster(G)        
            gc.load_curvature(gc, filename='Fan_' + str(k))
            gc.run_clustering(cluster_tpe='modularity_signed', cluster_by='curvature')
            gc.save_clustering(gc, filename='Fan_' + str(k))  
            
            gc.run_clustering(cluster_tpe='threshold', cluster_by='curvature')
            gc.save_clustering(gc, filename='Fan_' + str(k))
            
            gc.run_clustering(cluster_tpe='continuous_normalized', cluster_by='weight')
            gc.save_clustering(gc, filename='Fan_' + str(k))
            
            gc.run_clustering(cluster_tpe='linearized', cluster_by='weight')
            gc.save_clustering(gc, filename='Fan_' + str(k))
#            misc.plot_graph_snapshots(gc, node_labels= False, cluster=True)

        os.system('cd ..')             