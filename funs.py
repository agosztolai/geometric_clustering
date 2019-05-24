import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from multiprocessing import Pool
from functools import partial
import ot

# =============================================================================
# Geodesic distance matrix
# =============================================================================
#
# All pair shortest path using Floyd-Warshall algorithm
#     Input
#         An NxN NumPy array describing the directed distances between N nodes.
#         A[i,j] = adjacency matrix
#     Output
#         An NxN NumPy array such that result[i,j] is the geodesic distance 
#         between node i and node j. If i /~ i then result[i,j] == numpy.inf
# 
# =============================================================================
def distGeo(A):
    
    (mat, n) = check_and_convert_A(A)

    for k in range(n):
        mat = np.minimum(mat, mat[np.newaxis,k,:] + mat[:,k,np.newaxis]) 

    return mat     

#check A matrix
def check_and_convert_A(A):
    mat = A.copy() #create copy

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows
    
    #change zero elements to inf and zero diagonals
    mat[mat==0] = 100000#np.inf
    np.fill_diagonal(mat, 0)
    
    assert (np.diagonal(mat) == 0.0).all()

    return (mat, n)

# =============================================================================
# Curvature matrix (parallelised)
# =============================================================================
def ORcurvAll_sparse(G, dist, T, cutoff, lamb, workers, GPU=0):

    L = sc.sparse.csc_matrix(nx.normalized_laplacian_matrix(G), dtype=np.float64)
    
    if GPU == 0:
        with Pool(processes = workers) as p_mx:  #initialise the parallel computation
            mx_all = list(tqdm(p_mx.imap(partial(mx_comp, L, T, cutoff), G.nodes()),\
                           total = len(G)))

        with Pool(processes = workers) as p_kappa:  #initialise the parallel computation
            K = list(tqdm(p_kappa.imap(partial(K_comp, mx_all, dist, lamb), G.edges()),\
                          total = len(G.edges)))
    elif GPU == 1:
        with Pool(processes = workers) as p_mx:  #initialise the parallel computation
            mx_all = list(tqdm(p_mx.imap(partial(mx_comp, L, T, cutoff), G.nodes()),\
                           total = len(G)))
            
        K = K_comp_gpu(G,T,mx_all,dist,lamb)       
        
    return K

# unit vector (return a delta initial condition)
def delta(i, n):

    p0 = np.zeros(n)
    p0[i] = 1.

    return p0

# all neighbourhood densities
def mx_comp(L, T, cutoff, i):
    N = np.shape(L)[0]

    mx_all = [] 
    Nx_all = []
    for t in T: 
        mx_tmp = sc.sparse.linalg.expm_multiply(-t*L, delta(i, N))
        Nx = np.argwhere(mx_tmp > (1-cutoff)*np.max(mx_tmp))
        mx_all.append(sc.sparse.lil_matrix(mx_tmp[Nx]/np.sum(mx_tmp[Nx])))
        Nx_all.append(Nx)

    return mx_all, Nx_all

# compute curvature for an edge ij
def K_comp(mx_all, dist, lamb, e):
    i = e[0]
    j = e[1]

    n = len(mx_all[0][0])
    K = np.zeros(n)
    for it in range(n):

        Nx = np.array(mx_all[i][1][it]).flatten()
        Ny = np.array(mx_all[j][1][it]).flatten()
        mx = mx_all[i][0][it].toarray().flatten()
        my = mx_all[j][0][it].toarray().flatten()

        dNxNy = dist[Nx,:][:,Ny].copy(order='C')

        if lamb != 0:
            W = ot.sinkhorn2(mx, my, dNxNy, lamb)
        elif lamb == 0: #classical sparse OT
            W = ot.emd2(mx, my, dNxNy) 
            
        K[it] = 1. - W/dist[i, j]  

    return K

def K_comp_gpu(G,T,mx_all, dist, lamb):
#    import cupy
    import ot.gpu
#    
#    #        ot.gpu.to_gpu(mx_all) 
##        ot.gpu.to_gpu(dist)
##    for q,t in enumerate(T):
#    K = np.zeros()
#    for i in G.nodes:
#        ni = [n for n in G.neighbors(i) if n>i]  
#        mt = mx_all[i][1][q]
#        for k in ni:
#            mt = np.hstack(mt,mx_all[k][1][q])                
#        
#        dNxNy = dist[Nx,:][:,Ny].copy(order='C')
#        W = ot.gpu.sinkhorn(mt[:,1], mt[:,2:], dNxNy, lamb)    
#        K = 1. - W/dist[i, ni]
        
#    return K

# =============================================================================
# Cluster
# =============================================================================
def cluster(G,sample,perturb):
    from scipy.sparse.csgraph import connected_components as conncomp
    
    Aold = nx.adjacency_matrix(G).toarray()
    K_tmp = np.array(nx.to_numpy_matrix(G, weight='kappa')) 

    # cluster (remove edges with negative curv and find conn comps)   
    mink = np.min(K_tmp)
    maxk = np.max(K_tmp)
    labels = np.zeros([Aold.shape[0],sample])
    thres = np.append(0.0, np.random.normal(0, perturb*(maxk-mink), sample-1))

    nComms = np.zeros(sample)
    for k in range(sample):
        ind = np.where(K_tmp<=thres[k])     
        A = Aold.copy()
        A[ind[0],ind[1]] = 0 #remove edges with -ve curv.       
        (nComms[k], labels[:,k]) = conncomp(csr_matrix(A, dtype=int), directed=False, return_labels=True) 

    return nComms, labels

# =============================================================================
# Variation of information
# =============================================================================
def varinfo(comms): 

    comms = comms.astype(int)
    # Select only unique partitions
    (comms,ib,ic) = np.unique(comms,return_index=True,return_inverse=True,axis=1)
    
    # If all the partitions are identical, vi=0
    if len(ib) == 1:
        return 0, np.zeros([comms.shape[0], comms.shape[0]])   

    M = comms.shape[0]
    N = comms.shape[1] 
    vi_mat = np.zeros([M,M])
    
    vi_tot = 0  
    vi = 0
    nodes = list(range(N))

    # loop over all partition pairs once
    for i in range(M):
        partition_1 = comms[i,:]+1
        A_1 = csr_matrix((np.ones(len(nodes)), (partition_1,nodes)))
        n_1_all = np.array(np.sum(A_1,axis=1)).ravel()

        for k in range(i-1):
            partition_2 = comms[k,:]+1
            A_2 = csr_matrix((np.ones(len(partition_2)), (nodes,partition_2)))
            n_2_all = np.array(np.sum(A_2,axis=0)).ravel()
            n_12_all = A_1@A_2

            (rows,cols,n_12) = sc.sparse.find(n_12_all)

            n_1 = n_1_all[rows].ravel()
            n_2 = n_2_all[cols].ravel()

            vi = np.sum(n_12*np.log(n_12**2/(n_1*n_2)))
            vi = -1.0/(N*np.log(N))*vi
            vi_mat[i,k] = vi
            vi_tot = vi_tot + vi

    vi_mat_full = np.zeros([M,len(ic)])

    for i in range(M):
        vi_mat_full[i,:] = vi_mat[i,ic]

    vi_mat_full = vi_mat_full[ic,:]
    vi_mat = vi_mat_full + np.transpose(vi_mat_full)
    vi = np.mean(sc.spatial.distance.squareform(vi_mat))

    return vi, vi_mat

# =============================================================================
# Plot
# =============================================================================
def plotCluster(G,T,pos,t,comms,vi,nComms):
    import matplotlib.pyplot as plt
    import pylab
    import networkx as nx
    import os

    f = plt.figure(num=None, figsize=(10, 4), dpi=80, facecolor='w', edgecolor='k')
       
    # set edge colours and weights by curvature
    ax1 = plt.subplot(121)
    col = [G[u][v]['kappa'] for u,v in G.edges()]
    w = [G[u][v]['weight'] for u,v in G.edges()]    
    vmax = max([ abs(x) for x in col ])
    vmin = min([ abs(x) for x in col ])
    cmapedge = pylab.cm.bwr

    edges = nx.draw_networkx_edges(G,pos,edge_color=col,width=w, \
               edge_cmap=cmapedge, edge_vmin = -vmin, edge_vmax = vmax, ax=ax1)
        
    cbar = plt.colorbar(edges,ax=ax1)
    cbar.ax.set_ylabel('OR Curvature')
    
    ax1.set_aspect('equal', 'box')
    ax1.axis('off')
    
    # colour nodes by community
    cmapnode = plt.get_cmap("tab20")                    
    nodes = nx.draw_networkx_nodes(G, pos, node_color=comms, node_size=25, \
               cmap=cmapnode, with_labels=False, ax=ax1)
       
    # plot number of communities and VI
    ax2 = plt.subplot(122)
    ax3 = ax2.twinx()
    
    ax2.plot(T[0:len(nComms)], nComms, 'b-')
    ax3.plot(T[0:len(nComms)], vi, 'r-')
    
    ax2.set_xlabel('Markov time')
    ax2.set_ylabel('# communities', color='b')
    ax3.set_ylabel('Average variation of information', color='r')
    
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    
    ax2.set_xlim(T[0], T[-1])
    ax3.set_xlim(T[0], T[-1])

    ax2.set_ylim(0, len(G)+2)
    ax3.set_ylim(0, max(np.max(vi),0.01))
    
    ax2.tick_params('y', colors='b')
    ax3.tick_params('y', colors='r')  
    ax2.xaxis.grid(which="minor", color='k', linestyle='-', linewidth=0.5)
    ax2.yaxis.grid(which="minor", color='k', linestyle='-', linewidth=0.5)
       
    f.tight_layout()
    
    # Save
    if not os.path.isdir('images'):
        os.makedirs('images')
    
    plt.savefig('images/t_'+str(t)+'.png')
    
# =============================================================================
# Save
# =============================================================================
def savedata(filename,T,nComms,vi,data):
    import os
    
    if not os.path.isdir('data'):
            os.makedirs('data')
    f = open("data/"+filename+".dat","w") 
    f.write("T ")
    np.savetxt(f, T, fmt='%1.3f', delimiter=' ',newline=' ')
    f.write("\n")
    f.write("nComms ")
    np.savetxt(f, nComms.astype(int), fmt='%i',delimiter=' ',newline=' ')  
    f.write("\n")   
    f.write("vi ")
    np.savetxt(f, vi, fmt='%1.3f',delimiter=' ',newline=' ') 
    f.write("\n") 
    np.savetxt(f, data.astype(int), fmt='%i')
    f.close() 
    
# =============================================================================
# Load
# =============================================================================
def loaddata(filename,path = 0):
    import os

    if path == 0:
        path = os.getcwd()
        
    f = path + "/data/" + filename + ".dat"#path + "/" +  filename    
    f = open(f,"r")
    
    T = f.readline().split(' ')
    T = np.asarray(T[1:-1])
    nComms = f.readline().split(' ')
    nComms = np.asarray(nComms[1:-1])
    vi = f.readline().split(' ')
    vi = np.asarray(vi[1:-1])
    data = np.loadtxt(f)

    return T,nComms,vi,data    