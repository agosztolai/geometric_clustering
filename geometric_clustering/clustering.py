"""Clustering module."""
import networkx as nx
import numpy as np
import scipy.sparse as sp
from pygenstability import run
from pygenstability.constructors import Constructor


class constructor(Constructor):
    """Constructor for geometric modularity."""

    def __init__(self, *args, **kwargs):
        """ """

        self.kappas = None
        self.kappa0 = None
        self.time_dict = None
        self.row = None
        self.col = None

        super().__init__(*args, **kwargs)

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        self.kappas = kwargs["kappas"]
        self.kappa0 = kwargs["kappa0"]
        self.time_dict = kwargs["time_dict"]
        self.row = kwargs["row"]
        self.col = kwargs["col"]

    def get_data(self, time):
        """Return quality and null model at given time as well as global shift (or None)."""
        if self.kappa0 is None:
            # default is to ensure that at smallest time all edges are < 0 to have n_nodes clusters
            self.kappa0 = np.max(self.kappas[0]) * 1.01

        _kappas = np.array(self.kappas[self.time_dict[time]], dtype=np.float128)
        _kappas = (_kappas - self.kappa0) / (2 * np.sum(_kappas[_kappas > 0]))

        graph_kappa = sp.csr_matrix((_kappas, (self.row, self.col)), shape=self.graph.shape)
        self.partial_quality_matrix = graph_kappa + graph_kappa.T

        null_model = np.zeros(self.graph.shape[0])
        self.partial_null_model = np.array([null_model, null_model])

        return self.partial_quality_matrix, self.partial_null_model, None


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
    return run(
        graph=None,
        constructor=constructor(
            nx.adjacency_matrix(graph, weight="weight"),
            kappa0=kappa0,
            kappas=kappas,
            time_dict={time: i for i, time in enumerate(times)},
            row=np.array([e[0] for e in graph.edges]),
            col=np.array([e[1] for e in graph.edges]),
        ),
        times=times,
        n_louvain=n_louvain,
        with_VI=with_VI,
        n_louvain_VI=n_louvain_VI,
        with_postprocessing=with_postprocessing,
        with_ttprime=with_ttprime,
        n_workers=n_workers,
        tqdm_disable=tqdm_disable,
    )
