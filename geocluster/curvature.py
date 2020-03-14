"""Functions for computing the curvature"""
import multiprocessing
from tqdm import tqdm
import logging
import os
from time import time

import networkx as nx
import numpy as np
import scipy as sc
from scipy.sparse.csgraph import floyd_warshall
from sklearn.utils import check_symmetric
import ot

from .io import save_curvatures

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


class WorkerMeasures:
    """worker for building measures"""

    def __init__(self, laplacian, timestep):
        self.laplacian = laplacian
        self.timestep = timestep

    def __call__(self, measure):
        return heat_kernel(self.laplacian, self.timestep, measure)


class WorkerCurvatures:
    """worker for building measures"""

    def __init__(self, measures, geodesic_distances, params):
        self.measures = measures
        self.geodesic_distances = geodesic_distances
        self.params = params

    def __call__(self, edge):
        return edge_curvature(self.measures, self.geodesic_distances, self.params, edge)


def _get_chunksize(worker, graph, params, n_tries=10):
    """estimate good chunksize for POT parallel computation
       for fast POT computations, we will use large chunksize, 
       so little multiprocessing overhead, if POT computations are longer, 
       the chunksize will decrease until minimum value w.r.g n_workers"""
       
    if 'chunksize_time_step' not in params.keys():
        params["chunksize_time_step"] = 0.002
        
    if 'chunksize_time_min' not in params.keys():  
        params["chunksize_time_min"] = 0.004
        
    dtime_step = params["chunksize_time_step"]
    dtime_min = params["chunksize_time_min"]

    time0 = time()
    for i in range(n_tries):
        worker(list(graph.edges)[np.random.randint(len(graph.edges))])

    dtime = max(dtime_min, (time() - time0) / n_tries)

    chunksize = int(len(graph.edges) / ((dtime_step + dtime - dtime_min) / dtime_step))
    chunksize = max(chunksize, int(len(graph.edges) / params["n_workers"]))

    L.info(
        "Using chunksize = {}, (max={}, min={})".format(
            chunksize, len(graph.edges), int(len(graph.edges) / params["n_workers"])
        )
    )

    return chunksize


def compute_curvatures(graph, times, params, save=True, disable=False):
    """Edge curvature matrix"""

    L.info("Construct Laplacian")
    laplacian = construct_laplacian(graph, params["use_spectral_gap"])

    L.info("Compute geodesic distances")
    geodesic_distances = compute_distance_geodesic(graph)

    times_with_zero = np.hstack([0.0, times])  # add 0 timepoint

    kappas = np.ones([len(times), len(graph.edges())])
    measures = list(np.eye(len(graph)))
    pool = multiprocessing.Pool(params["n_workers"])

    ind = False
    L.info("Compute curvatures")
    for time_index in tqdm(range(len(times)), disable=True):
        L.info("---------------------------------")
        L.info("Step {}/{}".format(time_index, len(times)))
        L.info("Computing diffusion time 10^{:.1f}".format(np.log10(times[time_index])))

        worker_measure = WorkerMeasures(
            laplacian, times_with_zero[time_index + 1] - times_with_zero[time_index]
        )

        L.info("Computing measures")
        measures = pool.map(
            worker_measure,
            measures,
            chunksize=max(1, int(len(graph) / params["n_workers"])),
        )

        L.info("Computing curvatures")
        if not params["GPU"]:

            worker = WorkerCurvatures(measures, geodesic_distances, params)
            kappas[time_index] = pool.map(
                worker, graph.edges, chunksize=_get_chunksize(worker, graph, params)
            )

        else:
            for measure in measures:
                raise Exception("GPU code not working, WIP!")
                kappas[time_index] = edge_curvature_gpu(
                    measure,
                    geodesic_distances,
                    params["lamb"],
                    graph,
                    with_weights=params["with_weights"],
                )

        if all(kappas[time_index] > 0) and not disable and not ind:
            L.info(
                "All edges have positive curvatures, so you may stop the computations."
            )
            ind = True

        if save:
            save_curvatures(times[:time_index], kappas[:time_index])

    pool.close()

    return kappas


def construct_laplacian(graph, use_spectral_gap=True):
    """Laplacian matrix"""

    degrees = np.array([graph.degree[i] for i in graph.nodes])
    laplacian = nx.laplacian_matrix(graph).dot(sc.sparse.diags(1.0 / degrees))

    if use_spectral_gap:
        if len(graph) > 3:
            spectral_gap = abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0][1])
            L.info("Spectral gap = 10^{:.1f}".format(np.log10(spectral_gap)))
            laplacian /= spectral_gap
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
