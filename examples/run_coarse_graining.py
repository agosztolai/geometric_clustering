import sys as sys
import os
import numpy as np
import yaml

import matplotlib.pyplot as plt
import networkx as nx

import geocluster as gc 
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
times, kappas = io.load_curvature()

print('Compute scales')
edge_scales = gc.compute_scales(times, kappas)

print('Coarse grain')
graphs_reduc = gc.coarse_grain(graph, edge_scales, times)

print('plot coarse grain')
plotting.plot_coarse_grain(graphs_reduc, node_size=20, edge_width=1)
