'''main functionf for geoclusters'''
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import geocluster.curvature as curvature
import geocluster.io as io


def compute_curvatures(graph, times, params):
    """Edge curvature matrix"""

    laplacian = curvature.construct_laplacian(
        graph, params["laplacian_tpe"], params["use_spectral_gap"]
    )
    geodesic_distances = curvature.compute_distance_geodesic(graph)

    kappas = np.ones([len(times), len(graph.edges())])
    measures = list(np.eye(len(graph)))
    for time_index in tqdm(range(len(times) - 1)):
        with Pool(processes=params["n_workers"]) as p_mx:
            measures = p_mx.map_async(
                partial(
                    curvature.heat_kernel,
                    laplacian,
                    times[time_index + 1] - times[time_index],
                ),
                measures,
            ).get()

        if not params["GPU"]:
            with Pool(processes=params["n_workers"]) as p_kappa:
                kappas[time_index] = p_kappa.map_async(
                    partial(
                        curvature.edge_curvature, measures, geodesic_distances, params
                    ),
                    graph.edges(),
                ).get()
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

        if all(kappas[time_index] > 0):
            print(
                "All edges have positive curvatures, so you may stop the computations."
            )

        io.save_curvatures(times[:time_index], kappas[:time_index])

    return kappas

def compute_scales(times, kappas, method='zeros'):
    """compute the scales on edges, from curvatures"""
    if method == "zeros":
        edge_scales = []
        for kappa in kappas.T:
            crossing_id = np.argwhere(np.diff(np.sign(kappa)) == 2)
            if len(crossing_id) > 0:
                edge_scales += list(times[crossing_id[0]])
            else:
                edge_scales += [
                    times[0],
                ]

    elif method == "mins":
        edge_scales = times[np.argmin(kappas, axis=0)]
    return edge_scales


