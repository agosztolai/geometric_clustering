import sys as sys
sys.path.append('../utils')
import os as os
import yaml as yaml
from geometric_clustering import Geometric_Clustering
from graph_generator import generate_graph
from misc import save_curvature, plot_graph_snapshots
import numpy as np

#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#Load parameters
params = yaml.load(open('../utils/graph_params.yaml','rb'), Loader=yaml.FullLoader)[graph_tpe]
print('\nUsed parameters:', params)

#create a folder and move into it
if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)
        
#Load graph 
G = generate_graph(tpe=graph_tpe, params=params, save=True)
         
#Initialise the code with parameters and graph 
T = np.logspace(params['t_min'], params['t_max'], params['n_t'])
<<<<<<< HEAD
gc = Geometric_Clustering(G, T=T, cutoff=0.99, workers=16, GPU=False, lamb=0.)
=======
gc = Geometric_Clustering(G, T=T, cutoff=1., workers=16, GPU=False, lamb=0., laplacian_tpe='normalized')
>>>>>>> b676661b99fc7fe8ab58d404c78cf04b2b80a5b8

#Compute the OR curvatures
gc.compute_OR_curvatures()

#Save results for later analysis
plot_graph_snapshots(gc, node_labels= False, cluster=False)
save_curvature(gc)
