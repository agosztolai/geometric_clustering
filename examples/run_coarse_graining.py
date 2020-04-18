"""example of how to coarse grain a graph based on edge curvatures"""
import sys
import os
import yaml
import networkx as nx
import matplotlib.cm

import geocluster as gc
from geocluster import plotting, io
#from graph_library import generate

# get the graph from terminal input
graph_name = sys.argv[-1]

# load parameters
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]
params = yaml.full_load(open("params.yaml"))

#graph = generate(whichgraph=whichgraph, params=graph_params)
graph = nx.read_gpickle(os.path.join('graphs', 'graph_' + graph_name + '.gpickle'))

os.chdir(graph_name)

times, kappas = io.load_curvature()

print("Coarse grain")
graphs_reduc = gc.coarse_grain(graph, kappas, 0.5)

import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
eigs = []
lens = []
n_eig = int(0.05 * len(graph))
for graph in graphs_reduc:
    if len(graph) > n_eig:
        lens.append(len(graph))
        eigs.append(np.sort(np.linalg.eigh(1.*nx.laplacian_matrix(graph).toarray())[0])[:n_eig])

eigs = np.array(eigs)

plt.figure(figsize=(5,4))
cmap = matplotlib.cm.get_cmap('viridis')
for i, eig in enumerate(eigs.T[1:]):
    #plt.semilogx(times[:len(eig)], eig, c=str(i/len(eigs.T)))
    plt.plot(lens, eig, c=cmap(i/len(eigs.T)))

plt.xlabel('Size of coarse grained graph')
plt.ylabel('Eigenvalues')

#plt.twinx()
#plt.plot(lens)

plt.savefig('eigenvalues_coarse_grain.png')

plotting.plot_coarse_grain(graphs_reduc, node_size=20, edge_width=1)
