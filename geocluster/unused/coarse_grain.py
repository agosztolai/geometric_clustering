"""coarse grainning functions"""
import networkx as nx
import numpy as np
from tqdm import tqdm
import logging

from .scales import compute_scales
from .curvature import compute_curvatures

L = logging.getLogger(__name__)


def renormalize(full_graph, stepsize, params, threshold=-1e-7):
    """renormalize the graph with given stepsize"""

    graph = full_graph.copy()

    for u in graph:
        graph.nodes[u]["weight"] = 1.0

    graphs = [graph]
    while len(graph) > 1:
        L.info("Current graph size: {}".format(len(graph)))
        kappas = compute_curvatures(graph, [stepsize], params, save=False, disable=True)
        edge_scales = np.zeros(len(graph.edges))
        edge_scales[kappas[0] >= threshold] = -1.0
        graph = single_coarse_grain(graph, edge_scales, 0.0)
        graphs.append(graph)

    return graphs


def coarse_grain(graph, kappas, threshold):
    """coarse graain a graph for various thresholds"""
    graphs = []
    for kappa in tqdm(kappas):
        graphs.append(single_coarse_grain(graph, kappa, threshold))
    return graphs


def single_coarse_grain(graph, kappa, threshold):
    """coarse grain a graph at a single threshold"""

    # cut a copy of the graph
    graph_cut = graph.copy()
    for ei, e in enumerate(graph.edges()):
        if kappa[ei] < threshold:
            graph_cut.remove_edge(e[0], e[1])

    # find connected components and node to comp dict
    connected_comps = list(nx.connected_components(graph_cut))
    set_id = {u: i for u in graph for i, c in enumerate(connected_comps) if u in c}

    def _equivalence(u, v):
        """equivalence relation to quotient/coarse grain the graph"""
        if v in connected_comps[set_id[u]]:
            return True
        return False

    # quotient the graph
    graph_reduc = nx.quotient_graph(graph, _equivalence)

    # set position as mean of clustered nodes, and sum edges
    for u in graph_reduc:
        pos = []
        weight = 0
        for sub_u in u:
            pos.append(graph.nodes[sub_u]["pos"])
            for v in graph[sub_u]:
                if v in u:
                    if "weight" in graph[sub_u][v]:
                        weight += graph[sub_u][v]["weight"]
                    else:
                        weight += 1

        graph_reduc.nodes[u]["pos"] = np.array(pos).mean(0)
        graph_reduc.nodes[u]["weight"] = weight

    return nx.convert_node_labels_to_integers(graph_reduc)
