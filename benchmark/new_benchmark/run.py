import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

import geocluster as gc
from geocluster.io import save_curvatures
from pygenstability import plotting as plotting_pgs
from pygenstability.io import save_results

matplotlib.use("Agg")


if __name__ == "__main__":

    n_graphs = 10

    l = 2
    g = 500
    n = g * l
    k_mean = 3

    p_outs = np.linspace(0.1, 2.0, n_graphs) / n
    p_ins = 2 * k_mean / n - p_outs
    print(p_outs * n, p_ins * n, (p_ins - p_outs) * n)

    n_tries = 20

    seed = None

    n_t = 20
    t_min = -0.5
    t_max = 1.2

    for p_in, p_out in zip(p_ins, p_outs):
        print(p_in * n, p_out * n, (p_in - p_out) * n)
        lamb2 = 2 * p_out / (p_out + p_in)
        print("theoretical lambda2 = ", lamb2)

        diff = (p_in - p_out) * n

        for _try in range(n_tries):
            graph = nx.planted_partition_graph(l, g, p_in, p_out, seed=seed)
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = nx.convert_node_labels_to_integers(
                graph.subgraph(largest_cc).copy()
            )
            nx.write_gpickle(graph, f"graphs/graph_{p_out}_{_try}.gpickle")
            lapl = nx.laplacian_matrix(graph).toarray()
            w, v = np.linalg.eig(lapl)
            print("numerical lambda2= ", np.sort(w)[1])
            print("mean degree = ", np.mean([len(graph[node]) for node in graph.nodes]))

            blocks = [graph.nodes[node]["block"] for node in graph.nodes]
            times = np.logspace(t_min, t_max, n_t)
            kappas = gc.compute_curvatures(
                graph, times, n_workers=12, use_spectral_gap=False
            )
            save_curvatures(times, kappas, filename=f"kappas/kappas_{p_out}_{_try}")

            cluster_results = gc.cluster_signed_modularity(
                graph,
                times,
                kappas,
                kappa0=0,
                n_louvain=50,
                n_louvain_VI=10,
                n_workers=12,
                with_postprocessing=True,
                with_ttprime=False,
            )
            save_results(
                cluster_results, filename=f"clusters/cluster_{p_out}_{_try}.pkl"
            )
