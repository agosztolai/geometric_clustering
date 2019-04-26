def CurvCluster_parallel(G,pos,T,sample,cutoff,lamb=0,perturb=0.1,workers=1,filename=None,louvain=False, vis=0):

    import numpy as np
    from tqdm import tqdm
    from funs import distGeo, ORcurvAll_sparse_parallel, varinfo, plotCluster,savedata,cluster_louvain, cluster_threshold
    import networkx as nx
     
    A = nx.adjacency_matrix(G).toarray()   
    truth = []
    for i in G:
        truth.append(1)#G.node[i]['block'])
    for e, edge in enumerate(G.edges):
        G.edges[edge]['weight'] = A[edge]
    
# =============================================================================
#     geodesic distances (this is OK for directed)
# =============================================================================
    dist = np.array(distGeo(A))
    
# =============================================================================
#     compute curvature for all diffusion times
# =============================================================================    
    Kappa = ORcurvAll_sparse_parallel(G, dist, T, cutoff, lamb, workers)

# =============================================================================
#     process
# =============================================================================
    nComms = np.zeros([sample, len(T)]) 
    vi = np.zeros(len(T))
    data = np.empty([A.shape[0], len(T)+1])
    data[:,0] = truth
    for i in tqdm(range(len((T)))):

        # update edge curvatures
        for e, edge in enumerate(G.edges):
            G.edges[edge]['kappa'] = Kappa[e][i]            
            G.edges[edge]['weight'] = Kappa[e][i]            
    
        # cluster
        #nComms[:,i],labels = cluster(G,sample,perturb)
        if louvain:
            nComms[:,i], labels, vi[i] = cluster_louvain(G)
        else:
            nComms[:,i], labels, vi[i] = cluster_threshold(G,sample,perturb)
        
        # plot  
        if vis == 1:
            plotCluster(G,T,pos,i,labels,vi[0:i+1],nComms[0,0:i+1])
        
        # collect data to be saved
        data[:,i+1] = labels#[:,[0]]
        
# =============================================================================
#     save data
# =============================================================================
    if filename is not None:
        savedata(filename,T,nComms[0,:],vi,data)
