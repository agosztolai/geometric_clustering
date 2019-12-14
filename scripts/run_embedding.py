import sys as sys
sys.path.append('../utils')
import os as os
import geocluster
import networkx as nx
from geocluster.utils.misc import load_curvature, plot_graph_3D, plot_embedding
import yaml as yaml

#get the graph from terminal input 
graph_tpe = 'swiss-roll'#sys.argv[-1]
params = yaml.load(open('../utils/graph_params.yaml','rb'), Loader=yaml.FullLoader)[graph_tpe]

#load graph 
os.chdir(graph_tpe)
G = nx.read_gpickle(graph_tpe + ".gpickle")
         
# initialise the code with parameters and graph 
gc = geocluster(G)
 
#load results
load_curvature(gc)

#plot 3D graph
#for i in range(gc.Kappa.shape[1]):
#    params['counter']=i
#    plot_graph_3D(G, edge_colors=gc.Kappa[:,i], params=params, save=True)

#run embedding
#gc.run_embedding()

#plot embedding
#plot_embedding(gc)

#import sys
sys.path.append("../../utils") # Adds higher directory to python modules path.
#from embedding_utils import SpectralEmbedding
from sklearn.manifold import SpectralEmbedding
import pylab as plt
import numpy as np
n = G.number_of_nodes()
pos = nx.get_node_attributes(G, 'pos')
xyz = []
for i in range(len(pos)):
    xyz.append(pos[i])
xyz = np.array(xyz)
   
node_colors = nx.get_node_attributes(G, 'color')
colors = []
for i in range(n):
    colors.append(node_colors[i])
node_colors = np.array(colors)

for k in range(gc.Kappa.shape[1]):
#    se = SpectralEmbedding(n_components=2,affinity='nearest_neighbors', n_neighbors=10)
#    Y = se.fit_transform(xyz)
    
    A = np.zeros([n,n])
    for i,edge in enumerate(G.edges):
#        A[edge] = 1 
#        A[edge[::-1]] =1
        A[edge] = gc.Kappa[i,k]
        A[edge[::-1]] = gc.Kappa[i,k]
    
    se = SpectralEmbedding(n_components=2,affinity='precomputed')
    Y = se.fit_transform(A)

    plt.figure(figsize=(10,7))
    
    plt.scatter(Y[:, 0], Y[:, 1], c=colors)
    
    plt.axis('tight')
    plt.savefig(G.name +  str(k) + 'embedding.svg')