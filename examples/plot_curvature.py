import sys as sys
import os
import numpy as np
import yaml

from geocluster import plotting, io
from graph_library import generate

#get the graph from terminal input 
whichgraph = sys.argv[-1]     

#load parameters
paramsfile='graph_params.yaml'
params = yaml.load(open(paramsfile,'rb'), Loader=yaml.FullLoader)[whichgraph]

os.chdir(whichgraph)

#Load graph 
graph = generate(whichgraph=whichgraph, params=params)
         
#Compute the OR curvatures
times, curvatures = io.load_curvature()

#Save results for later analysis
plotting.plot_edge_curvatures(times, curvatures, ylog=True)
plotting.plot_graph_snapshots(graph, times, curvatures, folder='curvature_images', ext='.png')
