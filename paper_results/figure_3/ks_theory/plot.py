import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geocluster import curvature as cv
import scipy.sparse as sp
from tqdm import tqdm


if __name__ == "__main__":
    gs = [500, 5000, 50000]
    for g_ in gs:
        print(g_)
        plt.figure(figsize=(7,3))
        #c = next(cs)
        gt = np.ones(g_)
        gt[int(g_ / 2.)] = -1
        noises = []
        for _ in range(1000):
            r =  np.random.rand(g_)
            r[r<0.5] = -1.
            r[r>=0.5] = 1.
            noises.append(abs(np.dot(gt,r) / g_))

        #plt.axhline(np.mean(noises), c='k')
        #plt.axhline(np.mean(noises)-np.std(noises), c='k', ls='--')
        #plt.axhline(np.mean(noises)+np.std(noises), c='k', ls='--')
        results_df = pd.read_csv(f"results_{g_}.csv")
        std  = results_df.groupby('ks').std().reset_index()
        mean = results_df.groupby('ks').mean().reset_index()
        plt.errorbar(mean['ks'], mean['max_diff'], yerr=std['max_diff'],
                     label=f"diffs {2*g_}", c='C0')
        plt.errorbar(mean['ks'], mean['max_corr'], yerr=std['max_corr'],
                     label=f"corrs {2*g_}", ls='--', c='C1')
        plt.errorbar(mean['ks'], mean['lamb2'], yerr=std['lamb2'],
                     label=f"lamb2 {2*g_}", ls='--', c='C2')
        plt.axis([-0.05, mean['ks'].to_list()[-1], 0, 1.0])
        plt.legend()
        plt.savefig(f"ks_validation_{g_}.pdf")
        #plt.show()
