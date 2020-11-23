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
    with_VI=True,
    n_louvain_VI=10,
    with_postprocessing=True,
    with_ttprime=True,
    n_workers=1,
    tqdm_disable=False,
):
    """Cluster using signed modularity of Gomez, Jensen, Arenas PRE 2009.

    Args:
        graph (networkx): graph to cluster
        times (list): markov times to consider
        kappas (list): list of corresponding Kappa matrices
        kappa0 (float): shift of kappa via the null model
        n_louvain (int): number of Louvain evaluations
        with_VI (bool): compute the variatio of information between Louvain runs
        n_louvain_VI (int): number of randomly chosen Louvain run to estimate VI
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

        if kappa0 is None:
            # default is to ensure that at smallest time all edges are < 0 to have n_nodes clusters
            kappa0 = np.max(kappas[0]) * 1.01

        _kappas = np.array(kappas[time_dict[time]], dtype=np.float128)
        _kappas = (_kappas - kappa0) / (2 * np.sum(_kappas[_kappas > 0]))

        graph_kappa = sp.csr_matrix((_kappas, (row, cols)), shape=_graph.shape)
        quality_matrix = graph_kappa + graph_kappa.T
        null_model = np.zeros(len(graph.nodes))

        return quality_matrix, np.array([null_model, null_model])

    constructor = partial(modularity_constructor, kappa0=kappa0)

    return pgs.run(
        csgraph,
        constructor=constructor,
        times=times,
        n_louvain=n_louvain,
        with_VI=with_VI,
        n_louvain_VI=n_louvain_VI,
        with_postprocessing=with_postprocessing,
        with_ttprime=with_ttprime,
        n_workers=n_workers,
        tqdm_disable=tqdm_disable,
    )
