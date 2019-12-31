import sys as sys
import os as os

from geocluster import geocluster

from graph_library import graph_library as gl


#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#Load graph 
gg = gl.GraphGen(whichgraph=graph_tpe, paramsfile='./graph_params.yaml')
gg.generate()

#load graph 
os.chdir(graph_tpe)

# initialise the code with parameters and graph 
gc = geocluster.GeoCluster(gg.G)
 
#load results
gc.load_curvature()

#run embedding
gc.run_embeddings()

#plot embedding
gc.plot_embedding(folder='embedding_images')

