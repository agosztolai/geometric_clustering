import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geocluster import curvature as cv
import scipy.sparse as sp
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.cm as cm


def run(seed, l_, g_, k_in, k_out):
    graph = nx.planted_partition_graph(l_, g_, k_in / g_, k_out / g_, seed=seed)
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)
    graph = nx.convert_node_labels_to_integers(graph)

    laplacian = cv._construct_laplacian(graph, use_spectral_gap=False)
    w, v = sp.linalg.eigs(laplacian, k=100, which='SM')
    #w, v = np.linalg.eig(laplacian.toarray()) #, k=50, which='SM')

    C_1 = [node for node in graph.nodes if graph.nodes[node]["block"] == 0]
    C_2 = [node for node in graph.nodes if graph.nodes[node]["block"] == 1]
    edges = np.array(graph.edges)
    _e = edges[np.isin(edges[:, 0], C_1)]
    out_edges  = _e[np.isin(_e[:, 1], C_2)]

    gc = np.array([graph.nodes[node]["block"] for node in graph.nodes])
    gc[gc == 0] = -1
    diffs = []
    corr_int = []
    corr = []
    for s in range(len(w)):
        diffs.append(
            abs(np.sum([v[edge[0], s] - v[edge[1], s] for edge in out_edges]))
        )
        v_c_int = np.array(v[:, s])
        v_c_int[v_c_int < 0] = -1
        v_c_int[v_c_int > 0] = 1
        corr_int.append(abs(np.dot(gc, v_c_int))/ len(gc))
        corr.append(abs(np.dot(gc, v[:, s])/len(gc)))

    #w = w[1:]
    #v = v[:, 1:]
    #corr = corr[1:]
    #corr_int = corr_int[1:]
    #diffs = diffs[1:]

    v_c = v[:, np.argmax(diffs)]
    v_c_best = v[:, np.argmax(corr_int)]
    best_diff, best_corr = np.argmax(diffs), np.argmax(corr_int)

    v_c_int = np.array(v_c)
    v_c_int[v_c < 0] = -1
    v_c_int[v_c > 0] = 1
    v_c_int/= g_
    v_c_best_int = np.array(v_c_best)
    v_c_best_int[v_c_best < 0] = -1
    v_c_best_int[v_c_best > 0] = 1
    plot = True
    if plot:
        plt.figure(figsize=(4,3))
        norm_corr = np.array(corr_int)/ np.max(corr_int)
        plt.scatter(w, np.array(diffs)/max(diffs), s=10 + 20 * norm_corr, c=cm.turbo(norm_corr))#'k')#, marker=".")
        plt.savefig('diff_vs_eig.pdf', bbox_inches='tight')

        plt.figure(figsize=(4,3))
        plt.plot(w, corr_int, ".", c='k')
        plt.xlabel(r'$|\sum_{ij} (\phi_s(i) - \phi_s(j))\,\delta(C_i,C_j)|$')
        plt.ylabel(r'$corr$')

        plt.figure(figsize=(4,3))
        plt.scatter(diffs, corr_int, c=cm.turbo(norm_corr))
        # plt.scatter(diffs[best_diff], corr_int[best_diff], label="diff")
        plt.legend()
        plt.savefig('example_plot_1.pdf', bbox_inches='tight')

        plt.figure(figsize=(4,3))
        plt.plot(-v_c, ".", label="diff", c='k', ms=0.5)
        plt.axis([0, 10000, -.02, .02])
        #plt.plot(v_c_best, ".", label="corr")
        plt.xlabel('node id')
        plt.ylabel(r'$\phi_\mathrm{best}(i)$')
        plt.legend()
        # ax2 = plt.twinx()
        # ax2.set_xlim(0, 10000)
        #ax2.plot(v_c_int, ".", label="diff_int")
        #ax2.plot(0.8 * v_c_best_int, ".", label="corr_int")
        plt.plot(gc/np.sqrt(2*g_), c="r", lw=0.5)
        plt.savefig('best_eig.pdf', bbox_inches='tight')

        plt.figure(figsize=(4,3))
        plt.plot(v[:, 1], ".", label="diff", c='k', ms=0.5)
        plt.axis([0, 10000, -.02, .02])
        #plt.plot(v_c_best, ".", label="corr")
        plt.xlabel('node id')
        plt.ylabel(r'$\phi_2(i)$')
        plt.legend()
        # ax2 = plt.twinx()
        # ax2.set_xlim(0, 10000)
        #ax2.plot(v_c_int, ".", label="diff_int")
        #ax2.plot(0.8 * v_c_best_int, ".", label="corr_int")
        plt.plot(gc/np.sqrt(2*g_), c="r", lw=0.5)
        plt.savefig('second_eig.pdf')
    return corr_int[best_diff], corr_int[best_corr], corr_int[0]


if __name__ == "__main__":

    tau = 0.5
    l_ = 2
    g_ = 5000
    seed = 42
    k_out = 0.5
    k = 3.5
    k_in = k - k_out
    rks = (k - np.sqrt(k)) / (k + np.sqrt(k))
    print(rks - k_out/k_in)
    lambda2 = 2 * k_out / (k_in + k_out)

    max_diff, max_corr, lamb2 = run(seed, l_, g_, k_in, k_out)
