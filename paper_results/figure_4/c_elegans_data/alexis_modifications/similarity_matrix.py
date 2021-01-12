import pickle as pkl
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import networkx as nx
import sys
import numpy as np
from sklearn.decomposition import PCA
from RMST import RMST

# metrics = ['dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
#           'kulsinski', 'minkowski', 'rogerstanimoto',
#           'russellrao', 'sokalmichener', 'sokalsneath',
#           'yule']

metrics = ["jaccard"]
eps = 1
with_pca = False
with_rmst = True
gamma = 0.01

data = pkl.load(open("data/hox_gene_expression.pkl", "rb"))[:-1]
exp = data.iloc[:, 4:].to_numpy()

# remove genes that can be found in almost every neuron or no neuron
id1 = exp.sum(0) < exp.shape[0] * 0.9
id2 = exp.sum(0) > 0
exp = exp[:, id1 & id2]

ground_truth_1 = dict(enumerate(data["Neurotransmitter"]))
ground_truth_2 = dict(enumerate(data["Neuron Class"]))
neuron = data["Neuron"].to_list()


for m in metrics:

    dist = squareform(pdist(exp, metric=m))

    plt.figure()
    plt.imshow(dist)
    plt.colorbar()
    plt.savefig("distance_matrix.pdf")

    if with_pca:
        pca = PCA(n_components=20)
        principalComponents = pca.fit_transform(dist)
        dist = squareform(pdist(principalComponents, metric="euclidean"))
        print("pca explained variance:", sum(pca.explained_variance_ratio_))
        plt.figure()
        plt.imshow(dist)
        plt.colorbar()
        plt.savefig("distance_matrix_with_pca.pdf")

    similarity = 1.0 - dist
    #similarity = (np.max(dist) - dist) / np.max(dist)
    # similarity = np.exp(-dist/eps)
    similarity -= np.diag(np.diag(similarity))
    plt.figure()
    plt.imshow(similarity)
    plt.colorbar()
    plt.savefig("similarity_matrix.pdf")

    #    print(pca.singular_values_)
    G = nx.from_numpy_matrix(similarity)
    if with_rmst:
        G = RMST(G, gamma=gamma, weighted=True)
        plt.figure()
        plt.imshow(nx.adjacency_matrix(G).toarray())
        plt.colorbar()
        plt.savefig("sparse_adjacency.pdf")
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_size=1)
    nx.draw_networkx_edges(G, pos=pos, width=[G[u][v]['weight'] for u, v in G.edges])
    plt.savefig("graph.pdf")
    for i in G:
        G.nodes[i]["neurotransmitter"] = ground_truth_1[i]
        G.nodes[i]["neuron_class"] = ground_truth_2[i]
        G.nodes[i]["neuron"] = neuron[i]

    assert nx.is_connected(G), "graph is not connected"

    nx.write_gpickle(G, "data/hox_gene_expression_" + m + ".gpickle", protocol=4)
