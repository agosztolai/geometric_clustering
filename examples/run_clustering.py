"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle
import sys

import networkx as nx

from geocluster import cluster_signed_modularity, load_curvature

graph_name = sys.argv[-1]

graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

os.chdir(graph_name)

times, kappas = load_curvature()

cluster_results = cluster_signed_modularity(
    graph,
    times,
    kappas,
    kappa0=None,
    n_louvain=500,
    n_louvain_VI=50,
    n_workers=12,
    with_postprocessing=False,
)
pickle.dump(cluster_results, open("cluster_results.pkl", "wb"))
