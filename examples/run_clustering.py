"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle
import sys

import networkx as nx

from geocluster import cluster_signed_modularity, load_curvature

if __name__ == "__main__":
    graph_name = sys.argv[-1]

    graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
    graph = nx.convert_node_labels_to_integers(graph)

    times, kappas = load_curvature(filename=f"{graph_name}/curvature.pkl")
    times = times[:-8]
    kappas = kappas[:-8]

    cluster_results = cluster_signed_modularity(
        graph,
        times,
        kappas,
        kappa0=None,
        n_louvain=50,
        n_louvain_VI=50,
        n_workers=4,
        with_postprocessing=True,
    )
    pickle.dump(cluster_results, open(f"{graph_name}/cluster_results.pkl", "wb"))
