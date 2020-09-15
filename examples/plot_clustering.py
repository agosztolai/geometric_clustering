"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle
import sys

import networkx as nx

from geocluster import io
from geocluster.plotting import plot_communities
from pygenstability.plotting import plot_scan#, plot_single_community

graph_name = sys.argv[-1]

graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

os.chdir(graph_name)

times, kappas = io.load_curvature()

cluster_results = pickle.load(open("cluster_results.pkl", "rb"))

plot_scan(cluster_results, figure_name="figures/clustering_scan.svg", use_plotly=False)

plot_scan(cluster_results, figure_name="figures/clustering_scan.svg", use_plotly=True)

plot_communities(graph, kappas, cluster_results)
