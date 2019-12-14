def CurvCluster(G,L,pos,T,sample,cutoff,lamb,perturb,filename=None,louvain = False, vis=0):

    import numpy as np
    import scipy as sc
    from tqdm import tqdm
    from funs import distGeo, ORcurvAll_sparse, varinfo, plotCluster,savedata,cluster_louvain, cluster_threshold
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
            G.edges[edge]['weight'] = Kappa[edge[0]][edge[1]]
            
        # cluster
        #nComms[:,i],labels = cluster(G,sample,perturb)
        if louvain:
            nComms[:,i], labels, vi[i] = cluster_louvain(G)
        else:
            nComms[:,i], labels, vi[i] = cluster_threshold(G,sample,perturb)
 
        # compute VI
        #(vi[i],_) = varinfo(labels);
    
        # plot
        if vis == 1:
            plotCluster(G,T,pos,i,labels,vi[0:i+1],nComms[0,0:i+1])
    
        #collect data to be saved
        data[:,i+1] = labels#[:,[0]]

# =============================================================================
#     save data
# =============================================================================
    if filename is not None:
        np.savetxt('data/'+filename+'.dat',data)
