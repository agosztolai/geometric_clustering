import numpy as np
import networkx as nx
from pathlib import Path
import scipy as sc
import pylab as plt
from tqdm import tqdm
import matplotlib
import scipy.interpolate as sci
import networkx as nx
import pickle
import itertools

matplotlib.use("Agg")
marker = itertools.cycle(("p", "+", ".", "o", "*", "s", "v"))


def compute_separation(graph, kappa, edgelist):
    kappa_within = []
    kappa_between = []
    blocks = np.array([graph.nodes[node]["block"] for node in graph.nodes])
    for i, e in enumerate(edgelist):
        if blocks[e[0]] == 0 and blocks[e[1]] == 1:
            kappa_between.append(kappa[:, i])
        elif blocks[e[0]] == 1 and blocks[e[1]] == 0:
            kappa_between.append(kappa[:, i])
        else:
            kappa_within.append(kappa[:, i])

    kappa_within_mean = np.mean(kappa_within, axis=0)
    kappa_between_mean = np.mean(kappa_between, axis=0)
    kappa_within_var = np.var(kappa_within, axis=0)
    kappa_between_var = np.var(kappa_between, axis=0)

    # indx = np.where(kappa_within_mean >= 0.75)[0][0]
    indx = 0

    return abs(kappa_within_mean[indx] - kappa_between_mean[indx]) / np.sqrt(
        0.5 * (kappa_within_var[indx] + kappa_within_var[indx])
    )


if __name__ == "__main__":
    cases = 20
    c = [5, 8, 10, 15, 20, 25, 30]  # average degree
    n = [5000, 10000]
    th = 0.2
    times = [0.1]
    folder = Path("data")

    rs = []
    kappamin_avgs = []
    kappamin_stds = []
    rc2 = []
    for c_ in c:
        print(c_)
        c_in = np.linspace(c_ * 0.5, c_ * 0.9, cases)
        c_out = c_ - c_in
        r = c_out / c_in

        fname = folder / f"phase_transition_curvature_final_k_{c_}_{n[0]}.pkl"
        if not fname.exists():
            fname = folder / f"phase_transition_curvature_final_k_{c_}_{n[1]}.pkl"


        a = pickle.load(open(fname, "rb"))

        kappamin_avg = []
        kappamin_std = []
        zerocross_avg = []
        for case in tqdm(range(cases)):

            # select successful trials
            ntrials = len(a)
            graphs = [
                a[trial][case][0]
                for trial in range(ntrials)
                if not np.isnan(a[trial][case][1]).any()
            ]
            kappa = [
                a[trial][case][1]
                for trial in range(ntrials)
                if not np.isnan(a[trial][case][1]).any()
            ]
            edgelist = [
                a[trial][case][2]
                for trial in range(ntrials)
                if not np.isnan(a[trial][case][1]).any()
            ]

            kappamin = [
                compute_separation(graphs[i], kappa[i], edgelist[i])
                for i in range(len(kappa))
            ]
            # average over them
            kappamin_avg.append(np.mean(kappamin))
            kappamin_std.append(np.std(kappamin))

        # plot_phase_transition(r, kappamin_avg, kappamin_std, ax2)
        rs.append(r)
        kappamin_avgs.append(kappamin_avg)
        kappamin_stds.append(kappamin_std)

    pickle.dump(rs, open(folder / "rs.pkl", "wb"))
    pickle.dump(kappamin_avgs, open(folder / "kappamin_avgs.pkl", "wb"))
    pickle.dump(kappamin_stds, open(folder / "kappamin_stds.pkl", "wb"))
