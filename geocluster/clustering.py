"""Clustering module."""
import numpy as np
import scipy.sparse as sp
import networkx as nx
from functools import partial

try:
    from pygenstability import pygenstability as pgs
except ImportError:
    print("Pygenstability module not found, clustering will not work")


def cluster_signed_modularity(
    graph,
    times,
    kappas,
    kappa0=None,
    n_louvain=10,
    with_MI=True,
    n_louvain_MI=10,
    with_postprocessing=True,
    with_ttprime=True,
    n_workers=1,
    tqdm_disable=False,
):
    """Cluster using signed modularity of Gomez, Jensen, Arenas PRE 2009.

    Args:
        graph (networkx): graph to cluster
        global_time (float): scaling for the modularity to fix the
            global scale at thish modularity will work, similar to time
            in linearized markov stability.
        n_louvain (int): number of Louvain evaluations
        with_MI (bool): compute the mutual information between Louvain runs
        n_louvain_MI (int): number of randomly chosen Louvain run to estimate MI
        with_postprocessing (bool): apply the final postprocessing step
        with_ttprime (bool): compute the ttprime matrix
        n_workers (int): number of workers for multiprocessing
        tqdm_disable (bool): disable progress bars
    """
    time_dict = {time: i for i, time in enumerate(times)}
    csgraph = nx.adjacency_matrix(graph, weight="weight")

    def modularity_constructor(_graph, time, kappa0):
        """signed modularity contructor with curvature."""
        row = np.array([e[0] for e in graph.edges])
        cols = np.array([e[1] for e in graph.edges])
        graph_kappa = sp.csr_matrix(
            (kappas[time_dict[time]], (row, cols)), shape=_graph.shape
        )
        if kappa0 is None:
            kappa0 = np.mean(graph_kappa.data)

        quality_matrix = graph_kappa + graph_kappa.T
        null_model = kappa0 * np.ones(len(graph.nodes))

        return quality_matrix, np.array([null_model, null_model])

    constructor = partial(modularity_constructor, kappa0=kappa0)

    return pgs.run(
        csgraph,
        constructor=constructor,
        times=times,
        n_louvain=n_louvain,
        with_MI=with_MI,
        n_louvain_MI=n_louvain_MI,
        with_postprocessing=with_postprocessing,
        with_ttprime=with_ttprime,
        n_workers=n_workers,
        tqdm_disable=tqdm_disable,
    )
