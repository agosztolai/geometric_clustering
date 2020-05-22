"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from geocluster import io
from geocluster.plotting import plot_graph
from pygenstability.plotting import plot_scan, plot_single_community

graph_name = sys.argv[-1]

graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

os.chdir(graph_name)

times, kappas = io.load_curvature()

cluster_results = pickle.load(open("cluster_results.pkl", "rb"))

plot_scan(cluster_results, figure_name="figures/clustering_scan.svg", use_plotly=False)

plot_scan(cluster_results, figure_name="figures/clustering_scan.svg", use_plotly=True)


def plot_communities(
    graph,
    all_results,
    folder="communities",
    edge_color="0.5",
    edge_width=2,
    figsize=(15, 10),
):
    """now plot the community structures at each time in a folder"""
    if not os.path.isdir(folder):
        os.mkdir(folder)

    pos = [graph.nodes[u]["pos"] for u in graph]

    mpl_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    for time_id in tqdm(range(len(all_results["times"]))):
        plt.figure(figsize=figsize)
        plot_single_community(
            graph, all_results, time_id, edge_color="1", edge_width=3, node_size=10
        )
        plot_graph(
            graph, edge_color=kappas[time_id], node_size=0, edge_width=edge_width,
        )
        plt.savefig(
            os.path.join(folder, "time_" + str(time_id) + ".svg"), bbox_inches="tight"
        )
        plt.close()
    matplotlib.use(mpl_backend)


plot_communities(graph, cluster_results)
