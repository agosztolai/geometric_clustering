"""coarse grainning functions"""
import multiprocessing
from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

import geocluster.curvature as curvature
import geocluster.io as io


def coarse_grain(graph, edge_scales, thresholds):
    """coarse graain a graph for various thresholds"""
    graphs = []
    for threshold in tqdm(thresholds):
        graphs.append(single_coarse_grain(graph, edge_scales, threshold))
    return graphs


def single_coarse_grain(graph, edge_scales, threshold):
    """coarse grain a graph at a single threshold"""

    def _find_edge(graph_r):
        """find the next edge to coarse grain"""
        for e in graph_r.edges():
            if graph_r[e[0]][e[1]]['scale'] < threshold:
                return [e, ]
        return []

    graph_reduc = graph.copy()
    for ei, e in enumerate(graph_reduc.edges()):
        graph_reduc[e[0]][e[1]]['scale'] = edge_scales[ei]

    edgelist = _find_edge(graph_reduc)
    while len(edgelist) > 0:
        graph_reduc = nx.contracted_edge(graph_reduc, edgelist[0], self_loops=False)
        edgelist = _find_edge(graph_reduc)

    return nx.convert_node_labels_to_integers(graph_reduc)
