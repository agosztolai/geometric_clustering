import numpy as np
import scipy as sc
from tqdm import tqdm
import time as time
from scipy import optimize
import sys
from scipy.sparse import csc_matrix
import networkx as nx
from multiprocessing import Pool
from functools import partial




#--------------------------Curvature matrix
'''
Ollivier-Ricci curvature between two prob. measures mx(u) and my(v), 
supported on x and Nx = {u: u~x} and y and Ny = {v: v~y}, respectively. 
mx and my are the one-step transition probabilities of a lazy random walk, 
defined as mx(u) = alpha if u=x and (1-alpha)/dx if u~x and 0 otherwise.

INPUT: A adjacency matrix
       d distance matrix

OUTPUT: K NxN matrix with entries kij marking the curvature between
nodes i and j
'''

def delta(i, n):
    """
    return a delta initial condition
    """

    p0 = np.zeros(n)
    p0[i] = 1.

    return p0

def mx_comp(L, T, precision, i):
    mx_tmp = sc.sparse.linalg.expm_multiply(-L, delta(i, np.shape(L)[0]), T[0], T[-1], len(T) )
    mx_all = [] 
    for it in range(len(T)):
        mx_non_zero = np.where(mx_tmp[it] > precision)[0] 
        mx_all.append(sc.sparse.lil_matrix(mx_tmp[it, mx_non_zero]))

    return mx_all

def kappa_comp(mx_all, T, dist, e):
    # distribution at x and y supported by the neighbourhood Nx and Ny
    i = e[0]
    j = e[1]

    Kappa = np.zeros(len(T))
    for it, t in enumerate(T):

        mx_non_zero = mx_all[i][it].nonzero() 
        my_non_zero = mx_all[j][it].nonzero() 

        mx = mx_all[i][it][mx_non_zero].toarray()[0]
        my = mx_all[j][it][my_non_zero].toarray()[0]

        dist_xy = dist[mx_non_zero[1],:][:,my_non_zero[1]]

        W = W1(mx, my, dist_xy) #solve using simplex

        Kappa[it] = 1. - W/dist[i, j]  
    
    return Kappa


def ORcurvAll_sparse_full_parallel(G, dist, T, precision):

    N_n = len(G)
    N_e = len(G.edges)

    Kappa = np.zeros([len(T), N_n, N_n])

    L = sc.sparse.csc_matrix(nx.normalized_laplacian_matrix(G), dtype=np.float64)
    lamb_2 = np.max(abs(sc.sparse.linalg.eigs(L, which='SM', k=2)[0]))
    print('spectral gap:', lamb_2)
    L /= lamb_2


    n_processes = 5

    with Pool(processes = n_processes) as p_mx:  #initialise the parallel computation
        mx_all = list(tqdm(p_mx.imap(partial(mx_comp, L, T, precision), np.arange(N_n)), total = N_n))

    with Pool(processes = n_processes) as p_kappa:  #initialise the parallel computation
        Kappa = list(tqdm(p_kappa.imap(partial(kappa_comp, mx_all, T, dist), G.edges()), total = N_e))

    return Kappa

def ORcurvAll_sparse_full(A,dist,Phi,cutoff=1,lamb=0):

    #parse inputs
    eps = np.finfo(float).eps
    if cutoff == 0:
        cutoff = 1-eps

    N = A.shape[1]
    # loop over every edge once
    (x,y,_) = sc.sparse.find(A)
    KappaU = np.zeros([N,N])
    KappaL = np.zeros([N,N])
    for i in range(len(x)):
        # distribution at x and y supported by the neighbourhood Nx and Ny

        mx = Phi[x[i],:]
        my = Phi[y[i],:]

        mx_non_zero = np.where(mx>0)[0] 
        my_non_zero = np.where(my>0)[0] 

        mx = mx[mx_non_zero]
        my = my[my_non_zero]

        dist_xy = dist[mx_non_zero,:][:,my_non_zero]

        """
        if cutoff != 1:
            # Prune small masses to reduce problem size
            Nx = np.argsort(mx)[::-1]
            Ny = np.argsort(my)[::-1]  
            cmx = np.cumsum(mx[Nx])
            ind = cmx[1:] < cutoff
            Nx = Nx[np.insert(ind,0,True)] #always include first element
            mx = mx[Nx]/np.sum(mx[Nx])
            cmy = np.cumsum(my[Ny]) 
            ind = cmy[1:] < cutoff
            Ny = Ny[np.insert(ind,0,True)] 
            my = my[Ny]/np.sum(my[Ny])
            # Wasserstein distance between mx and my   
            dist = dist[Nx,:][:,Ny]
        """ 

        # curvature along x-y
        if lamb != 0: #entropy regularised OT
            print('does not work!')
            K = np.exp(-lamb*dist)
            mx = np.transpose(mx)
            my = np.transpose(my)
            (U,L) = sinkhornTransport(mx, my, K, K*dist, lamb)
            KappaL[x[i],y[i]] = 1 - U/dist[x[i],y[i]] 
            KappaU[x[i],y[i]] = 1 - L/dist[x[i],y[i]]  

        else: #classical sparse OT
            W = W1(mx, my, dist_xy) #solve using simplex
            KappaU[x[i],y[i]] = 1 - W/dist[x[i],y[i]]  
            #KappaL = KappaU

    KappaU = KappaU + np.transpose(KappaU)
    #KappaL = KappaL + np.transpose(KappaL)
    KappaU[np.abs(KappaU) < eps] = 0
    #KappaL[np.abs(KappaL) < eps] = 0
    
    return KappaU#, KappaU


#--------------------------Geodesic distance matrix
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


#--------------------------Diffusion 
def diffDist(L, t, retEval , dist = 0):
    import scipy.sparse.linalg as sla
    #import scipy.sparse as sp
    
    # compute diffusion
    if retEval == 0: # compute diffusion by matrix exponential
        Phi = sc.sparse.csr_matrix(sla.expm(-t*L.toarray()))
    else:  # compute diffusion by eigenvalue decomposition
        (evals, evecs) = sla.eigs(L, k=retEval, which='LA')
        #evals = np.diag(evals)
        Phi = np.transpose(evecs*np.exp(-np.real(evals)*t)) #diffusion map
        Phi = evecs@Phi
    
    eps = np.finfo(float).eps
    Phi[Phi < eps] = 0 
    
    N = L.shape[0]
    d = np.zeros([N, N])
    if dist == 1:      
        # all diffusion distances       
        for i in range(N):
            for j in range(i-1,N):
                d[i,j] = np.linalg.norm( Phi[:,i] - Phi[:,j] )

        d = d + d.T
        d[d<eps] = 0
    
    return Phi, d
                 
          
#--------------------------Exact optimal transport problem (lin program)  
''' Wasserstein distance (Hitchcock optimal transportation problem) between
 measures mx, my.
 beq is 1 x (m+n) vector [mx,my] of the one-step probability distributions
 mx and my at x and y, respectively
 d is 1 x m*n vector of distances between supp(mx) and supp(my)    
'''
def W1(mx, my, dist):

    nmx = len(mx)
    nmy = len(my)

    A1 = np.kron(np.ones(nmy), np.eye(nmx)) 
    A2 = np.kron( np.eye(nmy), np.ones(nmx))
    A = np.concatenate( (A1, A2), axis=0 )
    beq = np.concatenate((mx, my),axis=0)

    fval = optimize.linprog(dist.T.flatten(), A_eq=A, b_eq=beq,method='interior-point')#  method='simplex')
    #fval = optimize.linprog(dist.T.flatten(), A_eq=A, b_eq=beq,method='simplex')
       
    return fval.fun          

#--------------------------Entropy regularised optimal transport problem
'''     Compute N dual-Sinkhorn divergences (upper bound on the EMD) as well as
 N lower bounds on the EMD for all the pairs

---------------------------
 Inputs:
---------------------------
 a is either
    - a d1 x 1 column vector in the probability simplex (nonnegative,
    summing to one). This is the [1-vs-N mode]
    - a d_1 x N matrix, where each column vector is in the probability simplex
      This is the [N x 1-vs-1 mode]

 b is a d2 x N matrix of N vectors in the probability simplex

 K is a d1 x d2 matrix, equal to exp(-lamb M), where M is the d1 x d2
 matrix of pairwise distances between bins described in a and bins in the b_1,...b_N histograms.
 In the most simple case d_1=d_2 and M is simply a distance matrix (zero
 on the diagonal and such that m_ij < m_ik + m_kj


 U = K.*M is a d1 x d2 matrix, pre-stored to speed up the computation of
 the distances.

---------------------------
 Optional Inputs:
---------------------------
 stoppingCriterion in {'marginalDifference','distanceRelativeDecrease'}
   - marginalDifference (Default) : checks whether the difference between
              the marginals of the current optimal transport and the
              theoretical marginals set by a b_1,...,b_N are satisfied.
   - distanceRelativeDecrease : only focus on convergence of the vector
              of distances

 p_norm: parameter in {(1,+infty]} used to compute a stoppingCriterion statistic
 from N numbers (these N numbers might be the 1-norm of marginal
 differences or the vector of distances.

 tolerance : >0 number to test the stoppingCriterion.

 maxIter: maximal number of Sinkhorn fixed point iterations.
 
 verbose: verbose level. 0 by default.
---------------------------
 Outputs
---------------------------
 D : vector of N dual-sinkhorn divergences, or upper bounds to the EMD.

 L : vector of N lower bounds to the original OT problem, a.k.a EMD. This is computed by using
 the dual variables of the smoothed problem, which, when modified
 adequately, are feasible for the original (non-smoothed) OT dual problem

 u : d1 x N matrix of left scalings
 v : d2 x N matrix of right scalings

 The smoothed optimal transport between (a_i,b_i) can be recovered as
 T_i = diag(u(:,i)) * K * diag(v(:,i))

 or, equivalently and substantially faster:
 T_i = bsxfun(@times,v(:,i)',(bsxfun(@times,u(:,i),K)))

'''
    
def sinkhornTransport(mx,my,K,U,lamb,stoppingCriterion = 'marginalDifference',p_norm=np.inf,tolerance=0.005,maxIter=5000,VERBOSE=0):
    from numpy import transpose as tp
    
    # Checking the type of computation: 1-vs-N points or many pairs
    if mx.shape[1] == 1:
        ONE_VS_N = True # We are computing [D(mx,b_1), ... , D(mx,b_N)]
    elif mx.shape[1] == my.shape[1]:
        ONE_VS_N = False # We are computing [D(a_1,b_1), ... , D(a_N,b_N)]
    else:
        sys.exit('The first parameter mx is either a column vector in the probability simplex, or N column vectors in the probability simplex where N is size(my,2)')

    # Checking dimensionality
    if my.shape[1]>my.shape[0]:
        BIGN=True
    else:
        BIGN=False

    # Small changes in the 1-vs-N case to go a bit faster.
    if ONE_VS_N: # if computing 1-vs-N make sure all components of mx are >0. Otherwise we can get rid of some lines of K to go faster.
        I= (mx>0)
        if np.all(I) == 0: # need to update some vectors and matrices if mx does not have full support
            K=K[I,:]
            U=U[I,:]
            mx=mx[I]

        ainvK = np.divide(K,mx) # precomputation of this matrix saves a d1 x N Schur product at each iteration.

    # Fixed point counter
    compt=0

    # Initialization of Left scaling Factors, N column vectors.
    uL = np.divide(np.ones([mx.shape[0],my.shape[1]]), mx.shape[0])

    if stoppingCriterion == 'distanceRelativeDecrease':
        Dold=np.ones([1,my.shape[1]]) #initialization of vector of distances.

    # Fixed Point Loop
    # The computation below is mostly captured by the repeated iteration of uL=mx./(K*(my./(K'*uL)))
    while compt<maxIter:
        if ONE_VS_N: # 1-vs-N mode
            if BIGN:
                uL = np.divide(1.0, ainvK @ np.divide(my, tp(K)@uL)) # main iteration of Sinkhorn's algorithm
            else:
                uL = np.divide(1.0, ainvK @ np.divide(my, tp(tp(uL)@csc_matrix(K))))
        else: # N times 1-vs-1 mode
            if BIGN:
                uL = np.divide(mx, K @ np.divide(my, tp(tp(uL)@csc_matrix(K))))
            else:
                uL = np.divide(mx, K @ np.divide(my, tp(csc_matrix(K))@uL))

        compt=compt+1
    
        # check the stopping criterion every 20 fixed point iterations
        # or, if that's the case, before the final iteration to store the most
        # recent value for the matrix of right scaling factors vR.
        if np.mod(compt,20)==1 or compt==maxIter:   
            # split computations to recover right and left scalings.        
            if BIGN:
                vR = np.divide(my, tp(csc_matrix(K))@uL) # main iteration of Sinkhorn's algorithm
            else:
                vR = np.divide(my, tp(tp(uL)@csc_matrix(K)))
        
            if ONE_VS_N: # 1-vs-N mode
                uL= np.divide(1, ainvK@vR)
            else:
                uL = np.divide(mx, csc_matrix(K)@vR)       
                        
            # check stopping criterion
            if stoppingCriterion == 'distanceRelativeDecrease':
                D=np.sum(uL*U@vR)
                Criterion = np.linalg.norm(np.divide(D, Dold-1,p_norm))
                if Criterion<tolerance or np.isnan(Criterion):
                    break                
                Dold=D               
            elif stoppingCriterion == 'marginalDifference':
                Criterion = np.linalg.norm(np.sum(np.abs( np.multiply(vR,tp(K)@uL) - my )),p_norm)
                if Criterion<tolerance or np.isnan(Criterion): # npl.norm of all or . or_1 differences between the marginal of the current solution with the actual marginals.
                    break
            else:
                sys.exit('Stopping Criterion not recognized')        
      
            compt=compt+1
            if VERBOSE>0:
               print('Iteration :',compt,' Criterion: ',Criterion)        
        
            if np.any(np.isnan(Criterion)): # stop all computation if a computation of one of the pairs goes wrong.
                sys.exit('NaN values have appeared during the fixed point iteration. This problem appears because of insufficient machine precision when processing computations with a regularization value of lamb that is too high. Try again with a reduced regularization parameter lamb or with a thresholded metric matrix M.')

    if stoppingCriterion == 'marginalDifference': # if we have been watching marginal differences, we need to compute the vector of distances.
        D=np.sum(uL*U@vR)

    alpha = np.log(uL)
    beta = np.log(vR)
    beta[beta==-np.inf]=0 # zero values of vR (corresponding to zero values in my) generate inf numbers.
    if ONE_VS_N:
        L = (tp(mx)@alpha + np.sum(my*beta))/lamb
    else:       
        alpha[alpha==-np.inf]=0 # zero values of uL (corresponding to zero values in mx) generate inf numbers. in ONE-VS-ONE mode this never happens.
        L = (np.sum(mx*alpha) + np.sum(my*beta))/lamb
        
    return D, L

#--------------------------Plot
def plotCluster(G,pos,t,comms,numcomms):
    #import plotly.plotly as py
    #import plotly.graph_objs as go
    import pylab as plt
    import networkx as nx

    col = [G[u][v]['kappa'] for u,v in G.edges()]
    w = [G[u][v]['weight'] for u,v in G.edges()]
    plt.figure()
    
    # set edge colours and weights by curvature
    maxcol = max([ abs(x) for x in col ])
    mincol = min([ abs(x) for x in col ])
    nx.draw(G, pos, node_size=20, node_color='k',edge_color=col, width = w, edge_cmap = plt.cm.bwr, edge_vmin = -mincol, edge_vmax = maxcol)
    #nx.draw_networkx_edge_labels(G,pos,edge_labels=col)
    #print(np.min(col))
    #print(np.max(col))
    
    plt.savefig('images/t_'+str(t)+'.png')

    #plt.close() 
    #plt.figure()
    #plt.hist(col,bins=50)
    #plt.xlabel('curvature')
    #plt.savefig('images/hist_'+str(t)+'.png')
    
    
#    sp1 = subplot(1,2,1,'Parent',f)
#    if ~isempty(X) && ~isempty(Y):
#        p = plot(G,'XData',X,'YData',Y,'MarkerSize',4,'Parent',sp1)
#    else:
#        p = plot(G,'MarkerSize',6,'Parent',sp1) 
#
#    axis square
#
#    # set edge colours and weights by curvature
#    p.EdgeCData = G.Edges.Kappa #edge colour as curvature
#    p.LineWidth = G.Edges.Weight/max(G.Edges.Weight) #line width as weight 
#    labeledge(p,1:numedges(G),sign(G.Edges.Kappa))
#    cbar = colorbar
#    ylabel(cbar, 'OR curvature')
#    limit = max(np.abs(G.Edges.Kappa))
#    caxis([-limit, limit])
#
#    # colour nodes by community
#    if ~isempty(comms):
#    ColOrd = get(gca,'ColorOrder') m = size(ColOrd,1)
#    for i = 1:numnodes(G):
#        ColRow = rem(comms(i),m)
#        if ColRow == 0:
#            ColRow = m
#
#        Col = ColOrd(ColRow,:)
#        highlight(p,i,'NodeColor',Col)
#
#    sp2 = subplot(1,2,2,'Parent',f)
#    plot(T(1:length(N)),N,'Parent',sp2)
#    # set(ax(1),'YTickMode','auto','YTickLabelMode','auto','YMinorGrid','on')
#    # set(get(ax(1),'Ylabel'),'String','Number of communities')
#    set(sp2,'XLim', [10^floor(log10(T(1))) 10^ceil(log10(T(end)))], ...
#        'YLim', [1 10^ceil(log10(max(N)))], 'XScale','log','XMinorGrid','on')
#    set(sp2,'YScale','log')
#    xlabel('Markov time')
#    ylabel('Number of communities')
#
#
#    drawnow
#    frame = getframe(f) 
#    return frame      
    
#--------------------------Input graphs
def inputGraphs(n):
    import networkx as nx

    pos= []
    if n == 1:
        #Watts-Strogatz
        N = 20
        G = nx.newman_watts_strogatz_graph(N, 2, 0.30)  
        A = nx.to_numpy_matrix(G) 
        for i, j in G.edges():
            G[i][j]['weight'] = 1.
    
        #pos = nx.spring_layout(G)
        x = np.linspace(0,2*np.pi,N)
        posx = np.cos(x)
        posy = np.sin(x)
        
        for i in range(N):
            pos.append([posx[i],posy[i]])
            
    elif n == 2:
        #Symmetric barbell graph
        N = 10
        A = np.vstack((np.hstack((np.ones([N//2,N//2]), np.zeros([N//2,N//2]))), np.hstack((np.zeros([N//2,N//2]), np.ones([N//2,N//2])))))
        A = A-np.eye(N)
        A[N//2-1,N//2] = 1
        A[N//2,N//2-1] = 1
        G=nx.Graph(A)
        pos = nx.spring_layout(G)
        
    elif n == 3:
        A = np.matrix([[  0.,  86., 103., 139., 119.],
                       [ 86.,   0.,  95., 141.,  78.],
                       [103.,  95.,   0.,  37.,  59.],
                       [139., 141.,  37.,   0.,  72.],
                       [119.,  78.,  59.,  72.,   0.]])
        G=nx.Graph(A)
    
    
    # normalised Laplacian
    N = A.shape[0]
    A = sc.sparse.csr_matrix(A)
    diags = A.sum(axis=1).flatten()
    D = sc.sparse.spdiags(diags, [0], N, N, format='csr')
    L = D - A
    diags_sqrt = 1.0/sc.sqrt(diags)
    diags_sqrt[sc.isinf(diags_sqrt)] = 0
    DH = sc.sparse.spdiags(diags_sqrt, [0], N, N, format='csr')
    L = DH.dot(L.dot(DH))
    
    return G, A, L, pos
