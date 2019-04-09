import numpy as np
import scipy as sc
import time as time
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as conncomp
from funs import ORcurvAll_sparse, distGeo, plotCluster, inputGraphs, varinfo

# parameters
T = np.logspace(1, 2, 20) # diffusion time scale
#T = np.linspace(1e-4, 2, 2)  
retEval = 0                    # num. retained evals (currently not relevant)
cutoff = 0.99              # fraction of mass to retain
lamb = 0              # regularising parameter (the larger the more accurate, but higher cost, and too large can blow up)
whichGraph = 3
precision = 1e-10
sample = 10
perturb = 0.05

# load graph ##WARNING currently only works for directed
(G,Aold,L,pos) = inputGraphs(whichGraph) 
     
# geodesic distances (this is OK for directed)
dist = distGeo(Aold)

# loop over all diffusion times
nComms = np.zeros([sample, len(T)]) 
vi = np.zeros(len(T))
for i in tqdm(range(len((T)))):
    #print(' Diffusion time: ', t)
    
    A = Aold.copy()

    # compute diffusion after time t[i]
    Phi_full = sc.sparse.linalg.expm(-T[i]*L.toarray())
    Phi = (np.max(Phi_full)*precision)*np.round(Phi_full / (np.max(Phi_full)*precision))

    # compute curvatures
    Kappa = ORcurvAll_sparse(G.edges,dist,Phi,cutoff,lamb)    
    
    # update edge curvatures
    for edge in G.edges:
        G.edges[edge]['kappa'] = Kappa[edge[0]][edge[1]]

    #cluster (remove edges with negative curv and find conn comps)    
    mink = np.min(Kappa)
    maxk = np.max(Kappa)

    labels = np.zeros([A.shape[0],sample])
    thres = np.random.normal(0, perturb*(maxk-mink), sample)

    for k in range(sample):
        ind = np.where(Kappa<=thres[k])     
        A[ind[0],ind[1]] = 0 #remove edges with -ve curv.       
        (nComms[k,i], labels[:,k]) = conncomp(csr_matrix(A, dtype=int), directed=False, return_labels=True)
    
    # compute VI
    (vi[i],_) = varinfo(labels);
    
    # plot
    plotCluster(G,T,pos,i,labels[:,0],vi[0:i+1],np.mean(nComms[:,0:i+1],axis=0))