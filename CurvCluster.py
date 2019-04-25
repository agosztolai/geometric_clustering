def CurvCluster(G,L,pos,T,sample,cutoff,lamb,perturb,filename=None):

    import numpy as np
    import scipy as sc
    from tqdm import tqdm
    from funs import ORcurvAll_sparse, distGeo, plotCluster, varinfo, cluster
    import networkx as nx
    
    A = nx.adjacency_matrix(G).toarray()
    for e, edge in enumerate(G.edges):
        G.edges[edge]['weight'] = A[edge]
     
# =============================================================================
#     geodesic distances (this is OK for directed)
# =============================================================================
    dist = distGeo(A)

# =============================================================================
#     loop over all diffusion times
# =============================================================================
    nComms = np.zeros([sample, len(T)]) 
    vi = np.zeros(len(T))
    data = np.empty([A.shape[0], len(T)])
    for i in tqdm(range(len((T)))):
    
        # compute diffusion after time t[i]
        Phi_full = sc.sparse.linalg.expm(-T[i]*L.toarray())
        prec = 1e-10
        Phi = (np.max(Phi_full)*prec)*np.round(Phi_full / (np.max(Phi_full)*prec))

        # compute curvatures
        Kappa = ORcurvAll_sparse(G.edges,dist,Phi,cutoff,lamb)    
    
        # update edge curvatures
        for edge in G.edges:
            G.edges[edge]['kappa'] = Kappa[edge[0]][edge[1]]
            
        # cluster
        nComms[:,i],labels = cluster(G,sample,perturb)    
  
        # compute VI
        (vi[i],_) = varinfo(labels);
    
        # plot
        plotCluster(G,T,pos,i,labels[:,0],vi[0:i+1],np.mean(nComms[:,0:i+1],axis=0))
    
        #collect data to be saved
        data[:,[i]] = labels[:,[0]]

# =============================================================================
#     save data
# =============================================================================
    if filename is not None:
        np.savetxt('data/'+filename+'.dat',data)