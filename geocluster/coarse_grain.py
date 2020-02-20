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

    # cut a copy of the graph
    graph_cut = graph.copy()
    for ei, e in enumerate(graph.edges()):
        if edge_scales[ei] >= threshold:
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

    # set position as mean of clustered nodes
    for u in graph_reduc:
        pos = []
        for sub_u in u:
            pos.append(graph.nodes[sub_u]["pos"])
        graph_reduc.nodes[u]["pos"] = np.array(pos).mean(0)

    return nx.convert_node_labels_to_integers(graph_reduc)
