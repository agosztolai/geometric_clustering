import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy import optimize
import sys
from scipy.sparse import csc_matrix, csr_matrix
import networkx as nx
from multiprocessing import Pool
from functools import partial

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
# Curvature matrix
# =============================================================================    
#
# Ollivier-Ricci curvature between two prob. measures mi(k) and mj(l), which
# are defined as mi(k) = {Phi}ik, where Phi = Phi(t) = expm(-t*L).
# 
# INPUT: E list of edges
#        d distance matrix
# 
# OUTPUT: KappaU NxN matrices with entries kij marking the upper bound on the 
# OR curvature between nodes i and j
#    
# =============================================================================
def ORcurvAll_sparse(E,dist,Phi,cutoff=1,lamb=0):

    N = Phi.shape[1]  
    KappaU = np.zeros([N,N])
    
    # loop over every edge
    for e in E:
        
        # distribution at x and y 
        mx = Phi[e[0],:]
        my = Phi[e[1],:]

        # reduce support of mx and my
        Nx = np.where(mx>(1-cutoff)*np.max(mx))[0] 
        Ny = np.where(my>(1-cutoff)*np.max(my))[0]     
        
        # restrict & renormalise 
        dNxNy = dist[Nx,:][:,Ny] 
        mx = mx[Nx]
        my = my[Ny]
        mx = mx/sum(mx)
        my = my/sum(my)

        # curvature along x-y
        if lamb != 0: #entropy regularised OT (faster)
            K = np.exp(-lamb*dNxNy)
            (_,L) = sinkhornTransport(mx, my, K, K*dNxNy, lamb)
            KappaU[e[0],e[1]] = 1. - L/dist[e[0],e[1]]  

        else: #classical sparse OT
            W = W1(mx, my, dNxNy)
            KappaU[e[0],e[1]] = 1. - W/dist[e[0],e[1]]  

    KappaU = KappaU + np.transpose(KappaU)
    
    return KappaU

# =============================================================================
# Curvature matrix (parallelised)
# =============================================================================
def ORcurvAll_sparse_parallel(G, dist, T, cutoff, lamb, workers):

    N_n = len(G)
    N_e = len(G.edges)

    L = sc.sparse.csc_matrix(nx.normalized_laplacian_matrix(G), dtype=np.float64)
    #lamb_2 = np.max(abs(sc.sparse.linalg.eigs(L, which='SM', k=2)[0]))
    #print('spectral gap:', lamb_2)
    #L /= lamb_2

    with Pool(processes = workers) as p_mx:  #initialise the parallel computation
        mx_all = list(tqdm(p_mx.imap(partial(mx_comp, L, T, cutoff), G.nodes()), total = N_n))

    with Pool(processes = workers) as p_kappa:  #initialise the parallel computation
        Kappa = list(tqdm(p_kappa.imap(partial(kappa_comp, mx_all, T, dist, lamb), G.edges()), total = N_e))
    return Kappa

# unit vector (return a delta initial condition)
def delta(i, n):

    p0 = np.zeros(n)
    p0[i] = 1.

    return p0

# all neighbourhood densities
def mx_comp(L, T, cutoff, i):
    N = np.shape(L)[0]
    #mx_tmp= sc.sparse.linalg.expm_multiply(-L, delta(i, N), T[0], T[-1], len(T) )
    mx_all = [] 
    Nx_all = []
    for it, t in enumerate(T): 
        mx_tmp = sc.sparse.linalg.expm_multiply(-t*L, delta(i, N))
        Nx = np.argwhere(mx_tmp > (1-cutoff)*np.max(mx_tmp))
        mx_all.append(sc.sparse.lil_matrix(mx_tmp[Nx]/np.sum(mx_tmp[Nx])))
        Nx_all.append(Nx)

    return mx_all, Nx_all

# compute curvature for an edge ij
def kappa_comp(mx_all, T, dist, lamb, e):
    i = e[0]
    j = e[1]

    Kappa = np.zeros(len(T))
    for it, t in enumerate(T):

        Nx = np.array(mx_all[i][1][it]).flatten()
        Ny = np.array(mx_all[j][1][it]).flatten()

        mx = mx_all[i][0][it].toarray().flatten()
        my = mx_all[j][0][it].toarray().flatten()

        dNxNy = dist[Nx,:][:,Ny]

        if lamb != 0: #entropy regularised OT (faster)
            K = np.exp(-lamb*dNxNy)
            (_,L) = sinkhornTransport(mx, my, K, K*dNxNy, lamb)
            Kappa[it] = 1. - L/dist[i,j]  

        else: #classical sparse OT
            W = W1(mx, my, dNxNy) 
            Kappa[it] = 1. - W/dist[i, j]  

    return Kappa
                      
# =============================================================================
#  Exact optimal transport problem (lin program)  
# =============================================================================  
#
#  Wasserstein distance (Hitchcock optimal transportation problem) between
#  measures mx, my.
#  beq is 1 x (m+n) vector [mx,my] of the one-step probability distributions
#  mx and my at x and y, respectively
#  d is 1 x m*n vector of distances between supp(mx) and supp(my)    
# =============================================================================
def W1(mx, my, dist):

    nmx = len(mx)
    nmy = len(my)

    A1 = np.kron(np.ones(nmy), np.eye(nmx)) 
    A2 = np.kron(np.eye(nmy), np.ones(nmx))
    A = np.concatenate((A1, A2), axis=0)
    beq = np.concatenate((mx, my),axis=0)

    fval = optimize.linprog(dist.T.flatten(), A_eq=A, b_eq=beq, method='interior-point')
       
    return fval.fun          

# =============================================================================
# Entropy regularised optimal transport problem
# =============================================================================
#
#  Compute N dual-Sinkhorn divergences (upper bound on the EMD) as well as
#  N lower bounds on the EMD for all the pairs
# 
#  INPUTS
# 
#  mx : d1 x 1 column vector in the probability simplex (nonnegative,
#     summing to one)
# 
#  my : d2 x 1 column vector in the probability simplex
# 
#  K is a d1 x d2 matrix, equal to exp(-lamb M), where M is the d1 x d2
#  matrix of pairwise distances between bins described in a and bins in b. 
#  In the most simple case d_1=d_2 and M is simply a distance matrix 
# (zero on the diagonal and such that m_ij < m_ik + m_kj
# 
#  U = K.*M is a d1 x d2 matrix, pre-stored to speed up the computation of
#  the distances.
# 
#  OPTIONAL
# 
#  stoppingCriterion in {'marginalDifference','distanceRelativeDecrease'}
#    - marginalDifference (Default) : checks whether the difference between
#               the marginals of the current optimal transport and the
#               theoretical marginals set by a b_1,...,b_N are satisfied.
#    - distanceRelativeDecrease : only focus on convergence of the vector
#               of distances
# 
# 
#  tolerance : >0 number to test the stoppingCriterion.
# 
#  maxIter : maximal number of Sinkhorn fixed point iterations.
#   
#  verbose : 0 by default    
# 
#  OUTPUTS
# 
#  D : vector of N dual-sinkhorn divergences, or upper bounds to the EMD.
# 
#  L : vector of N lower bounds to the original OT problem, a.k.a EMD. This is 
#  computed by using the dual variables of the smoothed problem, which, when 
#  modified adequately, are feasible for the original (non-smoothed) OT dual 
#  problem
# =============================================================================    
def sinkhornTransport(mx,my,K,U,lamb,tolerance=0.005,maxIter=5000,VERBOSE=0):
    
    from numpy import transpose as tp

    mx = mx[:,np.newaxis]
    my = my[:,np.newaxis]
    
    # If not all components of mx are >0 we can get rid of some lines of K to go faster.
    I = np.where(mx)[0]
    if np.all(I) == 0: 
        K = K[I,:]
        U = U[I,:]
        mx = mx[I]

    ainvK = np.divide(K,mx) # precomputation of this matrix saves a d1 x 1 Schur product at each iteration.

    # Initialization of Left scaling Factors, N column vectors.
    uL = np.ones([mx.shape[0],1]) / mx.shape[0]

    # Fixed Point Loop: repeated iteration of uL=mx./(K*(my./(K'*uL)))
    compt=0
    while compt<maxIter:

        compt += 1
        uL = np.divide(1.0, ainvK @ np.divide(my, tp(tp(uL)@csc_matrix(K))))

        # check the stopping criterion every 20 fixed point iterations
        # or, if that's the case, before the final iteration to store the most
        # recent value for the matrix of right scaling factors vR.
        if np.mod(compt,20)==1 or compt==maxIter:   
            
            # split computations to recover right and left scalings.        
            vR = np.divide(my, tp(tp(uL)@csc_matrix(K)))        
            uL= np.divide(1.0, ainvK@vR)

            # check stopping criterion              
            Criterion = np.sum(np.abs( np.multiply(vR,tp(csc_matrix(K))@uL) - my ))
            if Criterion<tolerance or np.isnan(Criterion): # npl.norm of all or . or_1 differences between the marginal of the current solution with the actual marginals.
                break       
      
            compt += 1
            if VERBOSE>0:
               print('Iteration :',compt,' Criterion: ',Criterion)        
        
            if np.any(np.isnan(Criterion)): 
                sys.exit('NaN values have appeared during the fixed point \
                         iteration. This problem appears because of \
                         insufficient machine precision when processing \
                         computations with a regularization value of lamb that \
                         is too high. Try again with a reduced regularization\
                         parameter lamb or with a thresholded metric matrix M.')

    D = np.sum(uL*U@vR)

    alpha = np.log(uL)
    beta = np.log(vR)
    beta[beta==-np.inf]=0 # zero values of vR (corresponding to zero values in my) generate inf numbers.

    L = (tp(mx)@alpha + np.sum(np.multiply(my,beta)))/lamb

    return D, L

# =============================================================================
# Cluster
# =============================================================================
def cluster(G,sample,perturb):
    from scipy.sparse.csgraph import connected_components as conncomp
    
    Aold = nx.adjacency_matrix(G).toarray()
    Kappa_tmp = np.array(nx.to_numpy_matrix(G, weight='kappa')) 

    # cluster (remove edges with negative curv and find conn comps)   
    mink = np.min(Kappa_tmp)
    maxk = np.max(Kappa_tmp)
    labels = np.zeros([Aold.shape[0],sample])
    thres = np.append(0.0, np.random.normal(0, perturb*(maxk-mink), sample-1))

    nComms = np.zeros(sample)
    for k in range(sample):
        ind = np.where(Kappa_tmp<=thres[k])     
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
#    plt.show()
    
# =============================================================================
# Save
# =============================================================================
def savedata(filename,T,nComms,vi,data):
    import os
    
    if not os.path.isdir('data'):
            os.makedirs('data')
    f = open("data/"+filename+".dat","w") 
    f.write("T ")
    np.savetxt(f, T, fmt='%1.3f',delimiter=' ',newline=' ')
    f.write("\n")
    f.write("nComms ")
    np.savetxt(f, nComms.astype(int), fmt='%i',delimiter=' ',newline=' ')  
    f.write("\n")   
    f.write("vi ")
    np.savetxt(f, vi, fmt='%1.3f',delimiter=' ',newline=' ') 
    f.write("\n") 
    np.savetxt(f, data.astype(int), fmt='%i')
    f.close() 