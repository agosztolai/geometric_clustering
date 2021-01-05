import numpy as np
#from sklearn.metrics.cluster import mutual_info_score as mi
from pygenstability.pygenstability import WorkerVI as mi
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

from geocluster.io import load_curvature
from pygenstability.io import load_results

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

    for p_in, p_out in zip(p_ins, p_outs):
        print(p_in * n, p_out * n, (p_in - p_out) * n)
        lamb2 = 2 * p_out / (p_out + p_in)
        print("theoretical lambda2 = ", lamb2)

        diff = (p_in - p_out) * n

        fig1 = plt.figure()
        ax1 = plt.gca()

        fig2 = plt.figure()
        ax2 = plt.gca()

        fig3 = plt.figure()
        ax3 = plt.gca()
        for _try in range(n_tries):
            graph = nx.read_gpickle(f"graphs/graph_{p_out}_{_try}.gpickle")
            lapl = nx.laplacian_matrix(graph).toarray()
            w, v = np.linalg.eig(lapl)
            print("numerical lambda2= ", np.sort(w)[1])
            print("mean degree = ", np.mean([len(graph[node]) for node in graph.nodes]))

            blocks = [graph.nodes[node]["block"] for node in graph.nodes]
            times, kappas = load_curvature(filename=f"kappas/kappas_{p_out}_{_try}")
            cluster_results = load_results(
                filename=f"clusters/cluster_{p_out}_{_try}.pkl"
            )

            mis = []
            for time, n_com, comm_id in zip(
                times,
                cluster_results["number_of_communities"],
                cluster_results["community_id"],
            ):
                mis.append(mi([comm_id, blocks])([0, 1]))
                #mis.append(mi(comm_id, blocks))
            ax1.plot(cluster_results["number_of_communities"], mis, "+")
            ax1.axhline(1.0, ls="--", c="k")

            ax2.semilogx(times, mis)
            ax2.axhline(1.0, ls="--", c="k")
            ax2.axvline(1.0 / lamb2, ls="-", c="r")
            ax2.axvline(1.5 / lamb2, ls="-", c="b")
            ax2.axhline(0.0, ls="--", c="k")
            #ax2.set_ylim(-0.1, 1.1)
            # plotting_pgs.plot_scan(cluster_results, figure_name='figures/clustering_scan.svg', use_plotly=False)
            # plt.title('n_com ' + str(all_results['number_of_communities']) + 'stability' + str(all_results['stability']))
            # plt.show()
        fig1.savefig("figures_mis/n_com_mis_" + str(diff) + ".png")
        fig2.savefig("figures_time/time_mis_" + str(diff) + ".png")
        fig3.savefig("figures/scan_" + str(diff) + ".png")
        plt.close("all")
