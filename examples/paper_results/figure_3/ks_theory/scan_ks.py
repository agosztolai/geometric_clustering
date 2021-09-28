import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geometric_clustering import curvature as cv
import scipy.sparse as sp
from tqdm import tqdm
from joblib import Parallel, delayed


def run(seed, l_, g_, k_in, k_out):
    graph = nx.planted_partition_graph(l_, g_, k_in / g_, k_out / g_, seed=seed)
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)
    graph = nx.convert_node_labels_to_integers(graph)

    laplacian = cv._construct_laplacian(graph, use_spectral_gap=False)
    w, v = sp.linalg.eigs(laplacian, k=20, which="SM")

    edges = np.array(graph.edges)
    gc = np.array([graph.nodes[node]["block"] for node in graph.nodes])
    gc[gc == 0] = -1
    diffs = []
    corr_int = []
    corr = []
    for s in range(len(w)):
        diffs.append(abs(np.mean([v[edge[0], s] - v[edge[1], s] for edge in edges])))
        v_c_int = np.array(v[:, s])
        v_c_int[v_c_int < 0] = -1
        v_c_int[v_c_int > 0] = 1
        corr_int.append(abs(np.dot(gc, v_c_int)) / len(gc))
        corr.append(abs(np.dot(gc, v[:, s]) / len(gc)))

    w = w[1:]
    v = v[:, 1:]
    corr = corr[1:]
    corr_int = corr_int[1:]
    diffs = diffs[1:]

    v_c = v[:, np.argmax(diffs)]
    v_c_best = v[:, np.argmax(corr_int)]
    best_diff, best_corr = np.argmax(diffs), np.argmax(corr_int)

    v_c_int = np.array(v_c)
    v_c_int[v_c < 0] = -1
    v_c_int[v_c > 0] = 1
    v_c_best_int = np.array(v_c_best)
    v_c_best_int[v_c_best < 0] = -1
    v_c_best_int[v_c_best > 0] = 1
    plot = False
    if plot:
        plt.figure()
        plt.plot(w, diffs, ".")

        plt.figure()
        # plt.plot(w, corr, '.')
        plt.plot(w, corr_int, ".")

        plt.figure()
        # plt.plot(diffs, corr, 'b.')
        plt.plot(diffs, corr_int, ".")
        plt.plot(diffs[best_corr], corr_int[best_corr], "ro", label="corr")
        plt.plot(diffs[best_diff], corr_int[best_diff], "go", label="diff")
        plt.legend()

        plt.figure()
        plt.plot(v_c, ".", label="diff")
        plt.plot(v_c_best, ".", label="corr")
        plt.legend()
        ax2 = plt.twinx()
        ax2.plot(v_c_int, ".", label="diff_int")
        ax2.plot(0.8 * v_c_best_int, ".", label="corr_int")
        ax2.plot(gc, c="r")
    return corr_int[best_diff], corr_int[best_corr], corr_int[0]


if __name__ == "__main__":

    tau = 0.5
    l_ = 2
    # gs = [500, 5000, 50000]
    gs = [5000]
    # gs = [500]
    n_tries = 100

    k_outs = np.linspace(0.02, 0.9, 20)
    results_df = pd.DataFrame()
    plt.figure()
    for g_ in gs:
        i = 0
        for k_out in k_outs:
            print(k_out)
            k = 3
            k_in = k - k_out
            rks = (k - np.sqrt(k)) / (k + np.sqrt(k))
            lambda2 = 2 * k_out / (k_in + k_out)

            results = Parallel(-1, verbose=10)(
                delayed(run)(seed, l_, g_, k_in, k_out) for seed in range(n_tries)
            )
            for result in results:
                max_diff, max_corr, lamb2 = result
                results_df.loc[i, "max_diff"] = max_diff
                results_df.loc[i, "max_corr"] = max_corr
                results_df.loc[i, "lamb2"] = lamb2
                results_df.loc[i, "ks"] = rks - k_out / k_in
                i += 1

        results_df.to_csv(f"results_{g_}.csv", index=False)
        std = results_df.groupby("ks").std().reset_index()
        mean = results_df.groupby("ks").mean().reset_index()
        print(mean)
        print(std)
        plt.errorbar(mean["ks"], mean["max_diff"], yerr=std["max_diff"], label=f"diffs {g_}")
        plt.errorbar(mean["ks"], mean["max_corr"], yerr=std["max_corr"], label=f"corrs {g_}")
        plt.errorbar(mean["ks"], mean["lamb2"], yerr=std["lamb2"], label=f"lamb2 {g_}")
        plt.legend()
        plt.savefig("ks_validation.pdf")
        # plt.show()
