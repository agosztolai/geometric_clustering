"""Functions for computing the curvature"""
import networkx as nx
import numpy as np
import scipy as sc
from scipy.sparse.csgraph import floyd_warshall
from sklearn.utils import check_symmetric

import ot


class WorkerMeasures:
    """worker for building measures"""

    def __init__(self, laplacian, timestep):
        self.laplacian = laplacian
        self.timestep = timestep

    def __call__(self, measure):
        return curvature.heat_kernel(self.laplacian, self.timestep, measure)


class WorkerCurvatures:
    """worker for building measures"""

    def __init__(self, measures, geodesic_distances, params):
        self.measures = measures
        self.geodesic_distances = geodesic_distances
        self.params = params

    def __call__(self, edge):
        return curvature.edge_curvature(
            self.measures, self.geodesic_distances, self.params, edge
        )


def compute_curvatures(graph, times, params, save=True, disable=False):
    """Edge curvature matrix"""

    laplacian = curvature.construct_laplacian(graph, params["use_spectral_gap"])
    geodesic_distances = curvature.compute_distance_geodesic(graph)

    times_with_zero = np.hstack([0.0, times])  # add 0 timepoint

    kappas = np.ones([len(times), len(graph.edges())])
    measures = list(np.eye(len(graph)))
    pool = multiprocessing.Pool(params["n_workers"])
    ind = False
    for time_index in tqdm(range(len(times) - 1), disable=disable):

        worker_measure = WorkerMeasures(
            laplacian, times_with_zero[time_index + 1] - times_with_zero[time_index]
        )
        measures = pool.map(worker_measure, measures)

        if not params["GPU"]:
            kappas[time_index] = pool.map(
                WorkerCurvatures(measures, geodesic_distances, params), graph.edges()
            )

        else:
            for measure in measures:
                Warning("GPU code not working, WIP!")
                kappas[time_index] = curvature.edge_curvature_gpu(
                    measure,
                    geodesic_distances,
                    params["lamb"],
                    graph,
                    with_weights=params["with_weights"],
                )

        if all(kappas[time_index] > 0) and not disable and not ind:
            print(
                "All edges have positive curvatures, so you may stop the computations."
            )
            ind = True

        if save:
            io.save_curvatures(times[:time_index], kappas[:time_index])

    return kappas


def construct_laplacian(graph, use_spectral_gap=True):
    """Laplacian matrix"""

    degrees = np.array([graph.degree[i] for i in graph.nodes])
    laplacian = nx.laplacian_matrix(graph).dot(sc.sparse.diags(1.0 / degrees))

    if use_spectral_gap:
        laplacian /= abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0][1])

    return laplacian


def compute_distance_geodesic(G):
    """Geodesic distance matrix"""

    A = check_symmetric(nx.adjacency_matrix(G, weight="weight"))
    dist = floyd_warshall(A, directed=True, unweighted=False)

    return dist


# compute all neighbourhood densities
def heat_kernel(laplacian, timestep, measure):
    """compute matrix exponential on a measure"""
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, measure)


def edge_curvature(measures, geodesic_distances, params, edge):
    """compute curvature for an edge ij"""

    # get the edge/nodes ids
    i = edge[0]
    j = edge[1]

    # get the measures
    m_x = measures[i]
    m_y = measures[j]

    # set reduce the sized with cutoffs
    Nx = np.where(m_x >= (1.0 - params["cutoff"]) * np.max(m_x))[0]
    Ny = np.where(m_y >= (1.0 - params["cutoff"]) * np.max(m_y))[0]

    distances_xy = geodesic_distances[np.ix_(Nx, Ny)]

    m_x = m_x[Nx]
    m_y = m_y[Ny]

    m_x /= m_x.sum()
    m_y /= m_y.sum()

    # compute K
    if params["lambda"] != 0:  # entropy regularized OT
        wasserstein_distance = ot.sinkhorn2(m_x, m_y, distances_xy, params["lambda"])

    elif params["lambda"] == 0:  # classical sparse OT
        wasserstein_distance = ot.emd2(m_x, m_y, distances_xy)

    if params["with_weights"]:
        kappa = geodesic_distances[i, j] - wasserstein_distance
    else:
        kappa = 1.0 - wasserstein_distance / geodesic_distances[i, j]

    return kappa


def edge_curvature_gpu(mx_all, dist, lamb, G, with_weights=False):
    """compute curvature for an edge ij on gpu"""

    mx_all = ot.gpu.to_gpu(mx_all)
    dist = ot.gpu.to_gpu(dist.astype(float))
    lamb = ot.gpu.to_gpu(lamb)

    dist = dist.astype(float)

    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in x:
        ind = [y[1] for y in G.edges if y[0] == i]

        W = ot.gpu.sinkhorn(
            mx_all[:, i].tolist(), mx_all[:, ind].tolist(), dist.tolist(), lamb
        )
        if with_weights:
            Kt = np.append(Kt, ot.gpu.to_np(dist[i][ind] - W))
        else:
            Kt = np.append(Kt, 1.0 - W / ot.gpu.to_np(dist[i][ind]))

    return Kt
