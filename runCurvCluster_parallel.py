import numpy as np
import scipy as sc
import time as time
#from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as conncomp
#from funs import ORcurvAll_sparse_full, distGeo, diffDist, plotCluster, inputGraphs
from funs import *
#import networkx as nx

from params import *

# load graph ##WARNING currently only works for directed
try:
    G
except NameError:
    (G,Aold,L,pos) = inputGraphs(whichGraph) 
     
# geodesic distances (this is OK for directed)
dist = np.array(distGeo(Aold))


Kappa = ORcurvAll_sparse_parallel(G, dist, T, cutoff, lamb)

nComms = np.zeros([sample, len(T)]) 
vi = np.zeros(len(T))
for i in tqdm(range(len((T)))):
    #print(' Diffusion time: ', t)


    A = Aold.copy()
    # update edge curvatures
    for e, edge in enumerate(G.edges):
        #G.edges[edge]['weight'] = Kappa[e][i]
        G.edges[edge]['kappa'] = Kappa[e][i]

    #cluster (remove edges with negative curv and find conn comps)   

    Kappa_tmp = np.array(nx.to_numpy_matrix(G, weight='kappa')) 

    #cluster (remove edges with negative curv and find conn comps)   
    mink = np.min(Kappa_tmp)
    maxk = np.max(Kappa_tmp)
    labels = np.zeros([A.shape[0],sample])
    thres = np.random.normal(0, perturb*(maxk-mink), sample)

    for k in range(sample):
        ind = np.where(Kappa_tmp<=thres[k])     
        A = Aold.copy()
        A[ind[0],ind[1]] = 0 #remove edges with -ve curv.       
        (nComms[k,i], labels[:,k]) = conncomp(csr_matrix(A, dtype=int), directed=False, return_labels=True)
    
    # compute VI
    (vi[i],_) = varinfo(labels);
    
    plotCluster(G,T,pos,i,labels[:,0],vi[0:i+1],np.mean(nComms[:,0:i+1],axis=0))

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
