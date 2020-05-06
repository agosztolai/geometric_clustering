"""example of how to cluster a graph based on edges curvatures"""
import sys
import os
import yaml
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

import geocluster as gc
from geocluster import io
from geocluster.plotting import plot_graph

from pygenstability import plotting

graph_name = sys.argv[-1]

# load parameters
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]
params = yaml.full_load(open("params.yaml"))
graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

# labels = [graph.nodes[u]['club'] for u in graph]
# node_color = np.array([0 if label=='Mr. Hi' else 1 for label in labels])

# plt.figure()
# plot_graph(
#        graph,
#        node_colors=node_color
#    )
# plt.savefig('clubs.png')
# plt.show()

os.chdir(graph_name)

times, kappas = io.load_curvature()

cluster_results = gc.cluster(graph, times, kappas, params)

plotting.plot_scan(
    cluster_results, figure_name="figures/clustering_scan.svg", use_plotly=False
)
plt.show()


def plot_communities(
    graph, all_results, folder="communities", edge_color="0.5", edge_width=2
):
    """now plot the community structures at each time in a folder"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    pos = [graph.nodes[u]["pos"] for u in graph]

    mpl_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    # kappas[kappas<0] = -1
    # kappas[kappas>0] = 1
    for time_id in tqdm(range(len(all_results["times"]))):
        plt.figure()
        plotting.plot_single_community(
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
