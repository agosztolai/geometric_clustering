"""example of how to compute curvature of edges"""
import sys
import os
import yaml
import numpy as np
import networkx as nx

import geocluster as gc
from geocluster import plotting

graph_name = sys.argv[-1]
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]
graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

if not os.path.isdir(graph_name):
    os.mkdir(graph_name)

params = yaml.full_load(open("params.yaml", "rb"))

os.chdir(graph_name)

times = np.logspace(graph_params["t_min"], graph_params["t_max"], graph_params["n_t"])

# Compute the OR curvatures
print("Computing curvatures...")
kappas = gc.compute_curvatures(graph, times, params)
