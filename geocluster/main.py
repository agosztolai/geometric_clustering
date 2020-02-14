"""main functionf for geoclusters"""
import multiprocessing

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

import geocluster.curvature as curvature
import geocluster.io as io


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


def compute_scales(times, kappas, method="zeros"):
    """compute the scales on edges, from curvatures"""
    if method == "zeros":
        edge_scales = []
        for kappa in kappas.T:
            zero = InterpolatedUnivariateSpline(np.log10(times), kappa, k=3).roots()
            if len(zero) == 0:
                edge_scales.append(times[0])
            else:
                edge_scales.append(10 ** zero[0])

    elif method == "zeros_fast":
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

    print(edge_scales)
    return edge_scales
