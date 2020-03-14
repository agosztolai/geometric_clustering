"""example of how to compute curvature of edges"""
import sys
import os
import yaml
import numpy as np
import networkx as nx

import geocluster as gc
from geocluster import plotting
#from graph_library import generate

graph_name = sys.argv[-1]

# load parameters
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]

#graph = generate(whichgraph=whichgraph, params=graph_params)
graph = nx.read_gpickle(os.path.join('graphs', 'graph_' + graph_name + '.gpickle'))

if not os.path.isdir(graph_name):
    os.mkdir(graph_name)

params = yaml.full_load(open("params.yaml", "rb"))

os.chdir(graph_name)

times = np.logspace(graph_params["t_min"], graph_params["t_max"], graph_params["n_t"])

# Compute the OR curvatures
kappas = gc.compute_curvatures(graph, times, params)

plotting.plot_edge_curvatures(times, kappas, ylog=True, filename="edge_curvature")

plotting.plot_graph_snapshots(
    graph, times, kappas, folder="curvature_images", ext=".png"
)
