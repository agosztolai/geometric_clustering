"""main functionf for geoclusters"""
import multiprocessing
from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

import geocluster.curvature as curvature
import geocluster.io as io


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

    return edge_scales
