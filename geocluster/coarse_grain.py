"""coarse grainning functions"""
import networkx as nx
import numpy as np
from tqdm import tqdm


def coarse_grain(graph, edge_scales, thresholds):
    """coarse graain a graph for various thresholds"""
    graphs = []
    for threshold in tqdm(thresholds):
        graphs.append(single_coarse_grain(graph, edge_scales, threshold))
    return graphs


def single_coarse_grain(graph, edge_scales, threshold):
    """coarse grain a graph at a single threshold"""

    for ei, e in enumerate(graph.edges()):
        graph[e[0]][e[1]]["scale"] = edge_scales[ei]

    # TODO: cache the shortest path computations for speed up on subsequent coarse grainings
    def _equivalence(u, v):
        return (
            nx.shortest_path_length(graph, u, v, weight="scale")
            / len(nx.shortest_path(graph, u, v, weight="scale"))
            < threshold
        )

    graph_reduc = nx.quotient_graph(graph, _equivalence)

    for u in graph_reduc:
        pos = []
        for uu in u:
            pos.append(graph.nodes[uu]["pos"])
        graph_reduc.nodes[u]["pos"] = np.array(pos).mean(0)

    return nx.convert_node_labels_to_integers(graph_reduc)
