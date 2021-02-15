"""plot the computed edge curvature"""
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx

from geometric_clustering import load_curvature, plotting


if __name__ == "__main__":
    graph_name = sys.argv[-1]
    graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))

    times, kappas = load_curvature(filename=f"{graph_name}/curvature.pkl")

    plotting.plot_edge_curvatures(times, kappas, folder=graph_name, ext=".pdf")
    plotting.plot_edge_curvature_variance(times, kappas, folder=graph_name, ext=".pdf")
    plt.show()

    plotting.plot_graph_snapshots(
        graph, times, kappas, folder=f"{graph_name}/curvature_images", ext=".pdf", figsize=(12, 7)
    )
