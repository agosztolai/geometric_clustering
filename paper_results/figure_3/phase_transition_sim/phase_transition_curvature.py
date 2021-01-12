from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
import geocluster as gc
import pickle
import os
import networkx as nx


def generate_GN(params={"l": 4, "g": 32, "p_in": 0.4, "p_out": 0.2}, seed=0):

    if seed is not None:
        params["seed"] = seed

    G = nx.planted_partition_graph(
        params["l"], params["g"], params["p_in"], params["p_out"], seed=params["seed"]
    )

    labels_gt = []
    for i in range(params["l"]):
        labels_gt = np.append(labels_gt, i * np.ones(params["g"]))

    for n in G.nodes:
        G.nodes[n]["block"] = labels_gt[n - 1]

    return G, None


if __name__ == "__main__":

    cases = 20
    n = 10000
    frac_edges = .1  # 0 0.5
    trials = 100
    times = [0.01]  # np.logspace(-2.0, 1.0, 25)
    folder = Path("data_high_res")

    c_ = int(sys.argv[1])

    c_in = np.linspace(c_ * 0.5, c_ * 0.9, cases)
    c_out = c_ - c_in

    p_in = 2 * c_in / n
    p_out = 2 * c_out / n

    fname = folder / f"phase_transition_curvature_final_k_{c_}_{n}.pkl"

    if os.path.isfile(fname):
        print("results exists, continuing...")
        kappas = pickle.load(open(fname, "rb"))
    else:
        kappas = []

    seed = 0
    kappas = []
    for j in range(trials):
        print("trial " + str(j))
        seed = j

        kappa = []
        for i in tqdm(range(cases)):
            params = {"l": 2, "g": int(n / 2), "p_in": p_in[i], "p_out": p_out[i]}
            graph, _ = generate_GN(params, seed=int(seed))
            seed += 1
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = nx.convert_node_labels_to_integers(
                graph.subgraph(largest_cc).copy()
            )
            _edges = np.array([edge for edge in graph.edges])
            n_edges = int(frac_edges * len(_edges))
            print(f"n_edges={n_edges}")
            edgelist = [
                tuple(_edges[i])
                for i in np.random.randint(0, len(graph.edges), n_edges)
            ]
            _kappa = gc.compute_curvatures(
                graph,
                times,
                use_spectral_gap=False,
                n_workers=70,
                disable_tqdm=True,
                edgelist=edgelist,
                measure_cutoff=1e-4,
            )
            kappa.append([graph, _kappa, edgelist])

        kappas.append(kappa)

        pickle.dump(kappas, open(fname, "wb"))
