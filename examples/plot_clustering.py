"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle
import sys

import networkx as nx
import matplotlib.pyplot as plt

from geocluster import load_curvature
from geocluster.plotting import plot_communities
from pygenstability.plotting import plot_scan

if __name__ == "__main__":

    graph_name = sys.argv[-1]

    graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
    graph = nx.convert_node_labels_to_integers(graph)

    times, kappas = load_curvature(filename=f"{graph_name}/curvature.pkl")
    times = times[:-5]
    kappas = kappas[:-5]

    cluster_results = pickle.load(open(f"{graph_name}/cluster_results.pkl", "rb"))

    # cluster_results.pop("stability")  # to not plot stability

    plt.figure(figsize=(5, 3))
    plot_scan(cluster_results, figure_name=f"{graph_name}/clustering_scan.pdf", use_plotly=False)

    plot_scan(
        cluster_results,
        figure_name=f"{graph_name}/clustering_scan.pdf",
        use_plotly=True,
        live=False,
        plotly_filename=f"{graph_name}/clustering_scan.html",
    )
    plot_communities(graph, kappas, cluster_results, ext=".pdf", folder=f"{graph_name}/communities")

    # plt.show()
