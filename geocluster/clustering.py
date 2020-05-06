"""clustering functions"""
import multiprocessing
from functools import lru_cache
import scipy.sparse as sp

import networkx as nx
import numpy as np
from tqdm import tqdm

try:
    from pygenstability import pygenstability as pgs
    from pygenstability.constructors import constructor_signed_modularity
except ImportError:
    print("Pygenstability module not found, clustering will not work")


def cluster(graph, times, kappas, params, global_time=1.0):
    """Main clustering function."""

    if params["clustering_mode"] == "threshold":
        return cluster_threshold(graph, times, kappas, params)

    if params["clustering_mode"] == "signed_modularity":
        return cluster_signed_modularity(
            graph, times, kappas, params, global_time=global_time
        )

    raise Exception("Clustering method not understood")


def cluster_signed_modularity(graph_nx, times, kappas, params, global_time=1.0):
    """Cluster using signed modularity of Gomez, Jensen, Arenas PRE 2009. 
    
    The global_time argument is a scaling for the modularity to fix the
    global scale at thish modularity will work, similar to time in linearized 
    markov stability."""

    def modularity_constructor(graph, time):
        """signed modularity contructor with curvature."""
        row = np.array([e[0] for e in graph_nx.edges])
        cols = np.array([e[1] for e in graph_nx.edges])
        graph_kappa = sp.csr_matrix((kappas[int(time)], (row, cols)), shape=graph.shape)
        graph_kappa += graph_kappa.T
        return constructor_signed_modularity(graph_kappa, global_time)[:2]

    params["min_time"] = 0
    params["max_time"] = len(times) - 1
    params["n_time"] = len(times)
    params["log_time"] = False
    csgraph = nx.adjacency_matrix(graph_nx, weight="weight")
    return pgs.run(csgraph, params, constructor_custom=modularity_constructor)


def cluster_threshold(graph, times, kappas, params):  # pylint: disable=too-many-locals
    """run clustering by thresholding (with noise to estimate the quality"""
    if params["n_workers"] == 1:
        mapper = map
    else:
        pool = multiprocessing.Pool(params["n_workers"])
        mapper = pool.map

    all_results = {
        "times": [],
        "community_id": [],
        "number_of_communities": [],
        "params": params,
    }

    for time, kappa in tqdm(zip(times, kappas), total=len(times)):

        thresholds = np.random.normal(
            0,
            params["perturb"] * (np.max(kappas) - np.min(kappas)),
            params["n_samples"] - 1,
        )
        thresholds = np.append(0.0, thresholds)

        community_ids = []
        for threshold in thresholds:
            graph_threshold = graph.copy()
            for ei, e in enumerate(graph.edges()):
                if kappa[ei] <= threshold:
                    graph_threshold.remove_edge(e[0], e[1])

            community_id = np.zeros(len(graph))
            for i, cp in enumerate(list(nx.connected_components(graph_threshold))):
                community_id[list(cp)] = i
            community_ids.append(community_id)

        all_results["times"].append(time)
        all_results["community_id"].append(community_ids[0])
        all_results["number_of_communities"].append(len(set(community_ids[0])))

        pgs.compute_mutual_information(
            community_ids, all_results, mapper, params["n_samples"]
        )

    pgs.io.save_results(all_results)

    return all_results
