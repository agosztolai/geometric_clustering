import numpy as np
#from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as conncomp
from funs import ORcurvAll_sparse_full, distGeo, diffDist, plotCluster, inputGraphs
#import networkx as nx

# parameters
t = np.logspace(-2, 3, 20)  # diffusion time scale
retEval = 0                    # num. retained evals (currently not relevant)
prec = 1              # fraction of mass to retain
lamb = 10              # regularising parameter (the larger the more accurate, but higher cost, and too large can blow up)
whichGraph = 2

# load graph ##WARNING currently only works for directed
try:
    G
except NameError:
    (G,Aold,L,pos) = inputGraphs(whichGraph) 
     
# geodesic distances (this is OK for directed)
dist = np.array(distGeo(Aold.todense()))

# loop over all diffusion times
numcomms = np.zeros(len(t)) #v = 0 
for i in range(len(t)):
    print(' Diffusion time: ', t[i])
    
    A = Aold.copy()

    # compute diffusion after time t[i]
    (Phi,_) = diffDist(L,t[i],retEval)

    # compute curvatures
    (KappaL,KappaU) = ORcurvAll_sparse_full(A,dist,Phi,prec,lamb)    
    
    # update edge curvatures
    for edge in G.edges:
        G.edges[edge]['kappa'] = KappaU[edge[0]][edge[1]]

    #cluster (remove edges with negative curv and find conn comps)   
    A[KappaU<=0] = 0
    (numcomms[i], comms) = conncomp(csgraph=A, directed=False, return_labels=True)
    
    # append frame to movie
    #frame = 
    plotCluster(G,pos,t,comms,numcomms[i])
    #if i < len(t):
    #    stopmov = 0
    #else: 
    #    stopmov = 1
        
    #v = createMovie(frame, v, stopmov)      