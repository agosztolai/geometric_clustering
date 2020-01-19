import sys as sys
import os as os
from geocluster import GeoCluster
import networkx as nx

#get the graph from terminal input 
whichgraph = sys.argv[-1]

#Load graph 
G = nx.read_gpickle(whichgraph + "_0_.gpickle")

#load graph 
os.chdir(whichgraph)

# initialise the code with parameters and graph 
gc = GeoCluster(G)
 
#load results
gc.load_curvature()

#run embedding
gc.run_embeddings()

#plot embedding
gc.plot_embedding(folder='embedding_images')

