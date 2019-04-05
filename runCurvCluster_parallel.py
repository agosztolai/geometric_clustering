import numpy as np
import scipy as sc
import time as time
#from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as conncomp
#from funs import ORcurvAll_sparse_full, distGeo, diffDist, plotCluster, inputGraphs
from funs import *
#import networkx as nx

# parameters
T = np.linspace(1e-4, 2, 100)  # diffusion time scale
retEval = 0                    # num. retained evals (currently not relevant)
prec = 1              # fraction of mass to retain
lamb = 0              # regularising parameter (the larger the more accurate, but higher cost, and too large can blow up)
whichGraph = 2
precision = 1e-10

# load graph ##WARNING currently only works for directed
try:
    G
except NameError:
    (G,A,L,pos) = inputGraphs(whichGraph) 
     
# geodesic distances (this is OK for directed)
dist = np.array(distGeo(A.todense()))


Kappa = ORcurvAll_sparse_full_parallel(G, dist, T, precision)

numcomms = np.zeros(len(T)) #v = 0 
for i in tqdm(range(len((T)))):
    #print(' Diffusion time: ', t)

    # update edge curvatures
    for e, edge in enumerate(G.edges):
        G.edges[edge]['kappa'] = Kappa[e][i]

    #cluster (remove edges with negative curv and find conn comps)   

    Kappa_A = nx.to_numpy_matrix(G) 
    A[Kappa_A<=0] = 0
    (numcomms[i], comms) = conncomp(csgraph=A, directed=False, return_labels=True)
    
    # append frame to movie
    #frame = 
    plotCluster(G,pos,i,comms,numcomms[i])
 
import sys as sys
sys.exit()
print(np.shape(dist))
# loop over all diffusion times
numcomms = np.zeros(len(T)) #v = 0 
for i,t in enumerate(T):
    print(' Diffusion time: ', t)
    
    A = Aold.copy()

    # compute diffusion after time t[i]
    #(Phi,_) = diffDist(L,t[i],retEval)
    Phi_full = sc.sparse.linalg.expm(-t*L.toarray())
    Phi = (np.max(Phi_full)*precision)*np.round(Phi_full / (np.max(Phi_full)*precision))


    # compute curvatures
    Kappa = ORcurvAll_sparse_full(A,dist,Phi,prec,lamb)    
    
    # update edge curvatures
    for edge in G.edges:
        G.edges[edge]['kappa'] = Kappa[edge[0]][edge[1]]

    #cluster (remove edges with negative curv and find conn comps)   
    A[Kappa<=0] = 0
    (numcomms[i], comms) = conncomp(csgraph=A, directed=False, return_labels=True)
    
    # append frame to movie
    #frame = 
    plotCluster(G,pos,i,comms,numcomms[i])
    #if i < len(t):
    #    stopmov = 0
    #else: 
    #    stopmov = 1
        
    #v = createMovie(frame, v, stopmov)      
