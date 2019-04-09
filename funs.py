import numpy as np
import scipy as sc
from tqdm import tqdm
import time as time
from scipy import optimize
import sys
from scipy.sparse import csc_matrix, csr_matrix
import networkx as nx
from multiprocessing import Pool
from functools import partial

'''
#--------------------------Geodesic distance matrix
'''
def distGeo(A):
    '''All pair shortest path using Floyd-Warshall algorithm
    Input
        An NxN NumPy array describing the directed distances between N nodes.
        A[i,j] = adjacency matrix
    Output
        An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
    '''
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
    mat[mat==0] = np.inf
    np.fill_diagonal(mat, 0)
    
    assert (np.diagonal(mat) == 0.0).all()

    return (mat, n)

'''
#--------------------------Curvature matrix
Ollivier-Ricci curvature between two prob. measures mi(k) and mj(l), which
are defined as mi(k) = {Phi}ik, where Phi = Phi(t) = expm(-t*L).

INPUT: E list of edges
       d distance matrix

OUTPUT: KappaU NxN matrices with entries kij marking the upper bound on the 
OR curvature between nodes i and j
'''

def ORcurvAll_sparse(E,dist,Phi,cutoff=1,lamb=0):

    eps = np.finfo(float).eps

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
    KappaU[np.abs(KappaU) < eps] = 0.
    
    return KappaU

'''
#--------------------------Curvature matrix (parallelised)
'''
def ORcurvAll_sparse_parallel(G, dist, T, cutoff):

    N_n = len(G)
    N_e = len(G.edges)

    Kappa = np.zeros([len(T), N_n, N_n])

    L = sc.sparse.csc_matrix(nx.normalized_laplacian_matrix(G), dtype=np.float64)
    lamb_2 = np.max(abs(sc.sparse.linalg.eigs(L, which='SM', k=2)[0]))
    print('spectral gap:', lamb_2)
    L /= lamb_2


    n_processes = 5

    with Pool(processes = n_processes) as p_mx:  #initialise the parallel computation
        mx_all = list(tqdm(p_mx.imap(partial(mx_comp, L, T, cutoff), np.arange(N_n)), total = N_n))

    with Pool(processes = n_processes) as p_kappa:  #initialise the parallel computation
        Kappa = list(tqdm(p_kappa.imap(partial(kappa_comp, mx_all, T, dist), G.edges()), total = N_e))

    return Kappa

# unit vector
def delta(i, n):
    """
    return a delta initial condition
    """

    p0 = np.zeros(n)
    p0[i] = 1.

    return p0

# all neighbourhood densities
def mx_comp(L, T, cutoff, i):
    N = np.shape(L)[0]
    mx_tmp = sc.sparse.linalg.expm_multiply(-L, delta(i, N), T[0], T[-1], len(T) )
    mx_all = [] 
    for it in range(len(T)): 
        Nx = np.where(mx_tmp[it] > (1-cutoff)*np.max(mx_tmp[it]))[0] 
        mx_all.append(sc.sparse.lil_matrix(mx_tmp[it, Nx]))

    return mx_all

# all xy curvatures
def kappa_comp(mx_all, T, dist, e):
    # distribution at x and y supported by the neighbourhood Nx and Ny
    i = e[0]
    j = e[1]

    Kappa = np.zeros(len(T))
    for it, t in enumerate(T):

        Nx = mx_all[i][it].nonzero() 
        Ny = mx_all[j][it].nonzero() 

        mx = mx_all[i][it][Nx].toarray()[0]
        my = mx_all[j][it][Ny].toarray()[0]

        dNxNy = dist[Nx[1],:][:,Ny[1]]

        W = W1(mx, my, dNxNy) 
        Kappa[it] = 1. - W/dist[i, j]  
        
#        K = np.exp(-lamb*dNxNy)
#        (_,L) = sinkhornTransport(mx, my, K, K*dNxNy, lamb)
#        KappaU[e[0],e[1]] = 1. - L/dist[i,e[1]] 
    
    return Kappa
              
'''          
#--------------------------Exact optimal transport problem (lin program)  
 Wasserstein distance (Hitchcock optimal transportation problem) between
 measures mx, my.
 beq is 1 x (m+n) vector [mx,my] of the one-step probability distributions
 mx and my at x and y, respectively
 d is 1 x m*n vector of distances between supp(mx) and supp(my)    
'''
def W1(mx, my, dist):

    nmx = len(mx)
    nmy = len(my)

    A1 = np.kron(np.ones(nmy), np.eye(nmx)) 
    A2 = np.kron(np.eye(nmy), np.ones(nmx))
    A = np.concatenate((A1, A2), axis=0)
    beq = np.concatenate((mx, my),axis=0)

    fval = optimize.linprog(dist.T.flatten(), A_eq=A, b_eq=beq,method='interior-point')#  method='simplex')
       
    return fval.fun          

''' 
#--------------------------Entropy regularised optimal transport problem
 Compute N dual-Sinkhorn divergences (upper bound on the EMD) as well as
 N lower bounds on the EMD for all the pairs

 INPUTS:

 a is either
    - a d1 x 1 column vector in the probability simplex (nonnegative,
    summing to one). This is the [1-vs-N mode]
    - a d_1 x N matrix, where each column vector is in the probability simplex
      This is the [N x 1-vs-1 mode]

 b is a d2 x N matrix of N vectors in the probability simplex

 K is a d1 x d2 matrix, equal to exp(-lamb M), where M is the d1 x d2
 matrix of pairwise distances between bins described in a and bins in the 
 b_1,...b_N histograms. In the most simple case d_1=d_2 and M is simply a 
 distance matrix (zero on the diagonal and such that m_ij < m_ik + m_kj


 U = K.*M is a d1 x d2 matrix, pre-stored to speed up the computation of
 the distances.

 OPTIONAL

 stoppingCriterion in {'marginalDifference','distanceRelativeDecrease'}
   - marginalDifference (Default) : checks whether the difference between
              the marginals of the current optimal transport and the
              theoretical marginals set by a b_1,...,b_N are satisfied.
   - distanceRelativeDecrease : only focus on convergence of the vector
              of distances

 p_norm: parameter in {(1,+infty]} used to compute a stoppingCriterion 
 statistic from N numbers (these N numbers might be the 1-norm of marginal
 differences or the vector of distances.

 tolerance : >0 number to test the stoppingCriterion.

 maxIter: maximal number of Sinkhorn fixed point iterations.
 
 verbose: 0 by default.

 OUTPUTS

 D : vector of N dual-sinkhorn divergences, or upper bounds to the EMD.

 L : vector of N lower bounds to the original OT problem, a.k.a EMD. This is 
 computed by using the dual variables of the smoothed problem, which, when 
 modified adequately, are feasible for the original (non-smoothed) OT dual 
 problem

 u : d1 x N matrix of left scalings
 v : d2 x N matrix of right scalings

'''
    
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

'''
#--------------------------Variation of information
'''
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

'''
#--------------------------Plot
'''
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
    cmapnode = plt.get_cmap("tab10")                    
    nodecol = [10 if np.remainder(comms[i],10) == 0 else np.remainder(comms[i],10) for i in range(nx.number_of_nodes(G))]    
    
    nodes = nx.draw_networkx_nodes(G,pos,node_color=nodecol, node_size=20, \
               node_cmap=cmapnode, with_labels=False, ax=ax1)
       
    # plot number of communities and VI
    ax2 = plt.subplot(122)
    ax3 = ax2.twinx()
    
    ax2.plot(T[0:len(nComms)], nComms, 'b-')
    ax3.plot(T[0:len(nComms)],vi, 'r-')
    
    ax2.set_xlabel('Markov time')
    ax2.set_ylabel('# communities', color='b')
    ax3.set_ylabel('Average variation of information', color='r')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    #    ax3.set_yscale('log')
    
    ax2.set_xlim(10**np.floor(np.log10(T[0])), 10**np.ceil(np.log10(T[len(T)-1])))
    ax2.set_xlim(1, 10**np.ceil(np.log10(np.max(nComms))))   
    ax3.set_ylim(0, np.max(vi)*1.1+0.01)
    
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
    
'''    
#--------------------------Input graphs
'''
def inputGraphs(n):
    import networkx as nx
  
    if n == 1: #Watts-Strogatz      
        N = 20
        G = nx.newman_watts_strogatz_graph(N, 2, 0.30)  
        A = nx.to_numpy_matrix(G) 
        for i, j in G.edges():
            G[i][j]['weight'] = 1.
    
        x = np.linspace(0,2*np.pi,N)
        posx = np.cos(x)
        posy = np.sin(x)
        
        pos= []
        for i in range(N):
            pos.append([posx[i],posy[i]])
            
    elif n == 2: #Symmetric barbell graph      
        N = 10
        A = np.vstack((np.hstack((np.ones([N//2,N//2]), np.zeros([N//2,N//2]))), np.hstack((np.zeros([N//2,N//2]), np.ones([N//2,N//2])))))
        A = A-np.eye(N)
        A[N//2-1,N//2] = 1; A[N//2,N//2-1] = 1
        G=nx.Graph(A)
        pos = nx.spring_layout(G)
        
    elif n == 3: #Triangle of triangles
        m = 1
        N = 3
        A = np.ones([N, N])-np.eye(N)
        A = np.kron(np.eye(N**m),A)
        A[2,3]=1; A[3,2]=1; A[1,6]=1; A[6,1]=1; A[4,8]=1; A[8,4]=1
        A = np.vstack((np.hstack((A, np.zeros([9, 9]))), np.hstack((np.zeros([9, 9]), A))))
        A[0,9]=1; A[9,0]=1
    
        G = nx.Graph(A); 
        pos = nx.spring_layout(G,iterations=1000)
    
    
    # normalised Laplacian
#    N = A.shape[0]
#    A = sc.sparse.csr_matrix(A)
#    diags = A.sum(axis=1).flatten()
#    D = sc.sparse.spdiags(diags, [0], N, N, format='csr')
#    L = D - A
#    diags_sqrt = 1.0/sc.sqrt(diags)
#    diags_sqrt[sc.isinf(diags_sqrt)] = 0
#    DH = sc.sparse.spdiags(diags_sqrt, [0], N, N, format='csr')
#    L = DH.dot(L.dot(DH))
    L = nx.normalized_laplacian_matrix(G)
    
    return G, A, L, pos
