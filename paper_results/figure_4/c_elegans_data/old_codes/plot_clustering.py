"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle

import networkx as nx

from geocluster import io
# from geocluster.plotting import plot_communities
from pygenstability.plotting import plot_scan

graph_name = 'jaccard'

graph = nx.read_gpickle(os.path.join("data","hox_gene_expression_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

os.chdir(graph_name)

times, kappas = io.load_curvature()

for method in ['geometric_modularity', 'markovstab', 'modularity']:
    cluster_results = pickle.load(open(method + "_results.pkl", "rb"))

    plot_scan(cluster_results, figure_name="figures/" + method +  ".svg", use_plotly=False)

    plot_scan(cluster_results, figure_name="figures/"+ method + ".svg", use_plotly=True)

#gt = nx.get_node_attributes(graph,'gt2')

#plot_communities(graph, kappas, cluster_results, ground_truth = gt)
