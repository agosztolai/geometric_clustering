"""compute embeddings based on curvature"""
import networkx as nx
import numpy as np
import scipy as sc

from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import _deterministic_vector_sign_flip

from tqdm import tqdm


def signed_laplacian(graph):
    """compute the signed laplacian"""

    abs_adjacency = np.abs(nx.adjacency_matrix(graph))

    inv_sqrt_degrees = 1.0 / np.sqrt(np.array(abs_adjacency.sum(axis=0)).flatten())
    inv_sqrt_degrees = sc.sparse.diags(inv_sqrt_degrees)
    return sc.sparse.diags(np.ones(len(graph))) - inv_sqrt_degrees.dot(
        abs_adjacency
    ).dot(inv_sqrt_degrees)


def compute_single_embedding(graph, kappa, n_components=2):
    """embedding based on curvature-signed Laplacian eigenmaps"""

    graph_embedding = graph.copy()
    for ei, e in enumerate(graph_embedding.edges()):
        graph_embedding[e[0]][e[1]]["weight"] = kappa[ei]

    laplacian = signed_laplacian(graph_embedding)

    laplacian *= -1
    v0 = np.random.uniform(-1, 1, laplacian.shape[0])
    _, eigs = eigsh(laplacian, k=n_components, sigma=1.0, which="LM", v0=v0)

    embeddings = eigs.T[n_components::-1]
    return _deterministic_vector_sign_flip(embeddings)


def compute_embeddings(graph, times, kappas, n_components=2):
    """embedding based on curvature-signed Laplacian eigenmaps for all times"""
    embeddings = []
    for t in tqdm(range(len(times))):
        for e, edge in enumerate(graph.edges):
            graph.edges[edge]["curvature"] = kappas[t, e]
        embeddings.append(
            compute_single_embedding(graph, kappas[t], n_components=n_components)
        )
    return embeddings
