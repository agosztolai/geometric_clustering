#import scipy as sp
#import numpy as np
from numpy import logspace, zeros, array
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as conncomp

from funs import ORcurvAll_sparse_full, distGeo, Diff, plotCluster


# parameters
t = logspace(-2, 3, 20)  # diffusion time scale
l = 4                    # num. retained evals (currently not relevant)
prec = 0.99              # fraction of mass to retain
lamb = 10                # regularising parameter (the larger the more accurate, but higher cost, and too large can blow up)
vis = 0                  # plot visible (does not work yet)

# load graph
#if not exist('G','var'):
 #   [G,A,X,Y] = inputGraphs(13) #graph
A = array([[ 0., 24, 24, 76, 26],
           [62,  0, 88, 46, 73],
           [79,  7,  0, 29, 55],
           [63, 95,  8,  0, 14],
           [93,  5,  4, 58,  0]])


#f = figure('Visible',vis,'Position',[100 100 1600 600])

# geodesic distances OK
dist = distGeo(A)

numcomms = zeros(len(t)) #v = 0 
for i in range(len(t)):
    print(' Diffusion time: ', t[i])

    # compute diffusion after time t[i] OK
    Phi = Diff(A,t[i],l)

    # compute curvatures
    (KappaL,KappaU) = ORcurvAll_sparse_full(A,dist,Phi,prec,lamb)
        
    # update edge weights and curvatures
    #G.Edges.Weight = nonzero(tril(A))
    #indnonzeros = where(tril(A)>0) #edges with positive weights may have 0 kappa
    #G.Edges.Kappa = K(indnonzeros)
    
    # cluster  
    #ind = where(G.Edges.Kappa <= 0)
    #G1 = rmedge(G,ind) #remove edges with small curvature
    #comms = conncomp(G1)
    #N[i] = max(comms)
    Aclust = A
    Aclust[KappaU<=0] = 0
    Aclust = csr_matrix(Aclust)
    (numcomms[i], comms) = conncomp(csgraph=Aclust, directed=False, return_labels=True)
    
    # append frame to movie
    #frame = 
    plotCluster#(G,t,N,comms,X,Y,f)
    #if i < len(t):
    #    stopmov = 0
    #else: 
    #    stopmov = 1
        
    #v = createMovie(frame, v, stopmov)      