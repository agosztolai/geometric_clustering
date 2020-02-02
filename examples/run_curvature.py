import sys as sys
import os
import numpy as np
import yaml
import geocluster as gc 
from geocluster import plotting
from graph_library import generate

#get the graph from terminal input 
whichgraph = sys.argv[-1]     

#load parameters
paramsfile='graph_params.yaml'
params = yaml.load(open(paramsfile, 'rb'), Loader=yaml.FullLoader)[whichgraph]

if not os.path.isdir(whichgraph):
    os.mkdir(whichgraph)

os.chdir(whichgraph)

#Load graph 
graph = generate(whichgraph=whichgraph, params=params)
         
#Initialise the code with parameters and graph 
times = np.logspace(params['t_min'], params['t_max'], params['n_t'] + 1)

params = {}
params['n_workers'] = 1
params['GPU'] = False
params['lambda'] = False
params['with_weights'] = False
params['cutoff'] = 1.
params['laplacian_tpe'] = 'normalized'
params['use_spectral_gap'] = True

#Compute the OR curvatures
kappas = gc.compute_curvatures(graph, times, params)

plotting.plot_edge_curvatures(times, kappas, ylog=True)
plotting.plot_graph_snapshots(graph, times, kappas, folder='curvature_images', ext='.png')
