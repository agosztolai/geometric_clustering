"""example of how to cluster a graph based on edges curvatures"""
import os
import pickle

import networkx as nx
from pygenstability import pygenstability as pgs
from geocluster import cluster_signed_modularity, load_curvature

m = 'jaccard'

graph = nx.read_gpickle(os.path.join("data","hox_gene_expression_" + m + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

os.chdir(m)

times, kappas = load_curvature()
"""
modularity_results = pgs.run(
        nx.adjacency_matrix(graph, weight="weight"),
        constructor='linearized',
        times=times,
        n_louvain=200,
        n_louvain_VI=50,
    )
pickle.dump(modularity_results, open("modularity_results.pkl", "wb"))
"""

markovstab_results = pgs.run(
        nx.adjacency_matrix(graph, weight="weight"),
        constructor='continuous_normalized',
        #constructor='continuous_combinatorial',
        times=times,
        n_louvain=200,
        n_louvain_VI=50,
    )
pickle.dump(markovstab_results, open("markovstab_results.pkl", "wb"))

geometric_modularity_results = cluster_signed_modularity(graph, times, kappas, kappa0=None,
                                                         n_louvain=200, n_workers=12,
                                                         n_louvain_VI=50,
                                                         with_postprocessing=True,
                                                         )
pickle.dump(geometric_modularity_results, open("geometric_modularity_results.pkl", "wb"))


