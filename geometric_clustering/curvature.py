"""Functions for computing the curvature."""
import logging
import multiprocessing
import os
from functools import partial

import networkx as nx
import numpy as np
import ot
import scipy as sc
import scipy.sparse.csgraph as scg
from tqdm import tqdm

from .io import save_curvatures

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def _construct_laplacian(graph, use_spectral_gap=True):
    """Laplacian matrix."""
    degrees = np.array([graph.degree(i, weight="weight") for i in graph.nodes])
    laplacian = nx.laplacian_matrix(graph).dot(sc.sparse.diags(1.0 / degrees))
    if use_spectral_gap and len(graph) > 3:
        spectral_gap = sorted(abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0]))[1]
        L.debug("Spectral gap = 10^{:.1f}".format(np.log10(spectral_gap)))
        laplacian /= spectral_gap
    return laplacian


def _compute_distance_geodesic(G):
    """Geodesic distance matrix."""
    return scg.floyd_warshall(
        nx.adjacency_matrix(G, weight="weight"), directed=True, unweighted=False
    )


def _heat_kernel(measure, laplacian, timestep):
    """Compute matrix exponential on a measure."""
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, measure)


def _edge_curvature(
    edge,
    measures,
    geodesic_distances,
    measure_cutoff=1e-6,
    sinkhorn_regularisation=0,
    weighted_curvature=False,
):
    """Compute curvature for an edge."""
    node_x, node_y = edge
    m_x, m_y = measures[node_x], measures[node_y]

    Nx = np.where(m_x >= measure_cutoff * np.max(m_x))[0]
    Ny = np.where(m_y >= measure_cutoff * np.max(m_y))[0]

    m_x, m_y = m_x[Nx], m_y[Ny]
    m_x /= m_x.sum()
    m_y /= m_y.sum()

    distances_xy = geodesic_distances[np.ix_(Nx, Ny)]

    if sinkhorn_regularisation > 0:
        wasserstein_distance = ot.bregman.sinkhorn2(m_x, m_y, distances_xy, sinkhorn_regularisation)[0]
    else:
        wasserstein_distance = ot.emd2(m_x, m_y, distances_xy)

    if weighted_curvature:
        return geodesic_distances[node_x, node_y] - wasserstein_distance
    return 1.0 - wasserstein_distance / geodesic_distances[node_x, node_y]


def _edge_curvature_gpu(G, measures, geodesic_distances, sinkhorn_regularisation):   
    import ot.gpu
    # from .sinkhorn_gpu import sinkhorn_knopp, get_gpu_memory
    
    # total_gpu_mem = get_gpu_memory()
    
    # n = len(G.nodes)
    # m = len(G.edges)
    # total_mem = measures.nbytes*m*m/n + geodesic_distances.nbytes
    
    # n_chunks = 2*int(np.ceil(total_mem / total_gpu_mem))
    # size_chunks = int(np.floor(m/n_chunks))
    
    #load stuff to GPU
    measures = ot.gpu.to_gpu(measures) 
    geodesic_distances = ot.gpu.to_gpu(geodesic_distances.astype(float))
        
    # # =============================================================================
    # #version 1 (loop over edges)
    # for ch in range(n_chunks):
        
    #     i = [e[0] for i, e in enumerate(G.edges) if i >= ch*size_chunks and i < (ch+1)*size_chunks] 
    #     j = [e[1] for i, e in enumerate(G.edges) if i >= ch*size_chunks and i < (ch+1)*size_chunks]

    #     W = sinkhorn_knopp(measures[:,i], 
    #                    measures[:,j], 
    #                    geodesic_distances, 
    #                    sinkhorn_regularisation)
        
    # return 1. - W/ot.gpu.to_np(geodesic_distances[i][j])

    # =============================================================================
    #version 2 (loop over nodes and return K between all neighbours)
    K = []
    x = np.unique([x[0] for x in G.edges])
    for i in tqdm(x):

        ind = [y[1] for y in G.edges if y[0] == i]  

        W = ot.gpu.sinkhorn(measures[:,i], 
                        measures[:,ind], 
                        geodesic_distances, 
                        sinkhorn_regularisation,
                        to_numpy=False, 
                        log=False)
        print('ok')
        K = np.append(K, 1. - W/geodesic_distances[i][ind])
        
    return K


def compute_curvatures(
    graph,
    times,
    n_workers=1,
    use_spectral_gap=False,
    measure_cutoff=1e-6,
    sinkhorn_regularisation=0,
    use_gpu=False,
    weighted_curvature=False,
    filename="curvature.pkl",
):
    """Computes the curvatures of edges.

    Args:
        graph (networkx graph): graph to consider
        times (array): array of times to compute curvature
        n_workers (int): number of workers for multiprocessing
        use_spectral_gap (bool): to normalise time by the spectral gap of laplacian
        measure_cutoff (float): cutoff of the measures, in [0, 1], with no cutoff at 0
        sinkhorn_regularisation (float): Sinkhorn regularisation, when 0, no sinkhorn is applied
        weighted_curvature (bool): if True, the curvature is multiplied by the original edge weight
        filename (str): pickle filename to save curvatures at each time step
    """
    # Check for self-loops
    if nx.number_of_selfloops(graph) > 0:
        raise Exception("A graph with self-loops will not work!")

    # Check for connectedness
    degrees = [graph.degree(n) for n in graph.nodes]
    assert ~(np.array(degrees) == 0).any(), "Graph is not connected!"

    L.debug("Construct Laplacian")
    laplacian = _construct_laplacian(graph, use_spectral_gap)

    L.debug("Compute geodesic distances")
    geodesic_distances = _compute_distance_geodesic(graph)

    times_with_zero = np.insert(times, 0, 0.0)

    kappas = np.zeros([len(times), len(graph.edges())])
    measures = list(np.eye(len(graph)))
    display_all_positive = False
    L.debug("Compute all curvatures")
    with multiprocessing.Pool(n_workers) as pool:
        chunksize = max(1, int(len(graph.edges) / n_workers))
        for time_index in tqdm(range(len(times))):

            save_curvatures(times[:time_index], kappas[:time_index], filename=filename)
            
            if all(kappas[time_index] > 0) and display_all_positive:
                L.info("All edges have positive curvatures, so you may stop the computations.")
                display_all_positive = False
                
            L.debug("---------------------------------")
            L.debug("Step %s / %s", str(time_index), str(len(times)))
            L.debug("Computing diffusion time 10^{:.1f}".format(np.log10(times[time_index])))

            L.debug("Computing measures")
            measures = pool.map(
                partial(
                    _heat_kernel,
                    laplacian=laplacian,
                    timestep=times_with_zero[time_index + 1] - times_with_zero[time_index],
                ),
                measures,
                chunksize=chunksize,
            )
            
            L.debug("Computing curvatures")
            if use_gpu:
                try:
                    measures = np.array(measures)
                    kappas[time_index] = _edge_curvature_gpu(graph, 
                                        measures, 
                                        geodesic_distances, 
                                        sinkhorn_regularisation)
                except:
                    L.warn('There is no GPU or Cupy is not installed properly. Continuing on CPU.')
                    use_gpu=False
                    
            if not use_gpu:
                kappas[time_index] = pool.map(
                    partial(
                        _edge_curvature,
                        measures=measures,
                        geodesic_distances=geodesic_distances,
                        measure_cutoff=measure_cutoff,
                        sinkhorn_regularisation=sinkhorn_regularisation,
                        weighted_curvature=weighted_curvature,
                        ),
                    graph.edges,
                    chunksize=chunksize,
                )

    return kappas


def compute_OR_curvature(
    graph,
    n_workers=1,
    measure_cutoff=0.0,
    sinkhorn_regularisation=0,
    weighted_curvature=False,
):
    """Compute the original OR curvature of Ollivier 2007.

    Args:
        graph (networkx graph): graph to consider
        n_workers (int): number of workers for multiprocessing
        measure_cutoff (float): cutoff of the measures, in [0, 1], with no cutoff at 0
        sinkhorn_regularisation (float): Sinkhorn regularisation value, when 0, no sinkhorn is used
        weighted_curvature (bool): if True, the curvature if multiplied by the original edge weight
    """
    L.debug("Construct transition matrix")
    adjacency = nx.adjacency_matrix(graph)
    inv_degree = sc.sparse.diags(1.0 / np.array(adjacency.sum(0)).flatten())
    transition_matrix = adjacency.dot(inv_degree)

    L.debug("Compute geodesic distances")
    geodesic_distances = _compute_distance_geodesic(graph)

    L.debug("Computing measures")
    measures = [transition_matrix.dot(measure) for measure in list(np.eye(len(graph)))]

    L.debug("Computing curvatures")
    with multiprocessing.Pool(n_workers) as pool:
        kappas = pool.map(
            partial(
                _edge_curvature,
                measures=measures,
                geodesic_distances=geodesic_distances,
                measure_cutoff=measure_cutoff,
                sinkhorn_regularisation=sinkhorn_regularisation,
                weighted_curvature=weighted_curvature,
            ),
            graph.edges,
            chunksize=max(1, int(len(graph.edges) / n_workers)),
        )

    return kappas
