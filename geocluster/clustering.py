"""clustering functions"""
import multiprocessing

import networkx as nx
import numpy as np
from tqdm import tqdm

try:
    from pygenstability import pygenstability as pgs
    from pygenstability.io import save
    from pygenstability.constructors import constructor_signed_modularity
    from pygenstability.constructors import constructor_continuous_linearized
except ImportError:
    print("Pygenstability module not found, clustering will not work")


def cluster(graph, times, kappas, params):
    """main clusterin function"""

    if params["clustering_mode"] == "threshold":
        return cluster_threshold(graph, times, kappas, params)

    if params["clustering_mode"] == "signed_modularity":
        return cluster_signed_modularity(graph, times, kappas, params)

    raise Exception("Clustering method not understood")


def cluster_signed_modularity(graph, times, kappas, params):
    """cluster usint signed mofularity of Gomez, Jensen, Arenas PRE 2009"""

    def modularity_constructor(graph, time):
        """signed modularity contructor with curvature"""
        graph_kappa = graph.copy()
        for ei, e in enumerate(graph_kappa.edges()):
            graph_kappa[e[0]][e[1]]["weight"] = kappas[int(time)][ei]
        return constructor_signed_modularity(graph_kappa, 1.0)

    params["min_time"] = 0
    params["max_time"] = len(times) - 1
    params["n_time"] = len(times)
    params["log_time"] = False
    params["save_qualities"] = False

    return pgs.run(graph, params, constructor=modularity_constructor)


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

    save(all_results)

    return all_results
