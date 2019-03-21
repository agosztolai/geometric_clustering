import numpy as np
from numpy import  where, triu, zeros, ones, eye, inf, kron, diag, asarray, minimum, diagonal, newaxis, divide, multiply, mod, isnan, log
from numpy import concatenate as cat
from numpy import transpose as tp
from scipy.linalg import expm
from numpy.linalg import norm
from numpy.linalg import multi_dot as matprod
from scipy.linalg import fractional_matrix_power as fracpow
from scipy.optimize import linprog
import sys
from scipy.sparse import csc_matrix


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
def ORcurvAll_sparse_full(A,dist,Phi,cutoff=0,lamb=0):

    #parse inputs
    eps = np.finfo(float).eps
    if cutoff == 0:
        cutoff = 1-eps

    N = A.shape[1]
    # loop over every edge once
    ind = where(triu(A) > 0)
    x = ind[0]
    y = ind[1]
    KappaU = zeros([N,N])
    KappaL = zeros([N,N])
    for i in range(len(x)):

        # distribution at x and y supported by the neighbourhood Nx and Ny
        mx = Phi[x[i],:]
        my = Phi[y[i],:]
    
        # Prune small masses to reduce problem size
        Nx = mx.argsort()[::-1]
        Ny = my.argsort()[::-1]  
        cmx = np.cumsum(mx[Nx])
        ind = cmx[1:] < cutoff
        Nx = Nx[np.insert(ind,0,True)] #always include first element
        mx = mx[Nx]/np.sum(mx[Nx])
        cmy = np.cumsum(my[Ny]) 
        ind = cmy[1:] < cutoff
        Ny = Ny[np.insert(ind,0,True)] 
        my = my[Ny]/np.sum(my[Ny])
    
        # Wasserstein distance between mx and my   
        dNxNy = dist[Nx,:][:,Ny]

        # curvature along x-y
        if lamb != 0: #entropy regularised OT
            K = np.exp(-lamb*dNxNy)
            mx = np.array(mx,ndmin=2)
            my = np.array(my,ndmin=2)
            (U,L) = sinkhornTransport(tp(mx),tp(my),K,multiply(K, dNxNy),lamb)
            KappaL[x[i],y[i]] = 1 - U/dist[x[i],y[i]] 
            KappaU[x[i],y[i]] = 1 - L/dist[x[i],y[i]]  
        else: #classical sparse OT
            W = W1(mx,my,dNxNy) #solve using simplex
            KappaU[x[i],y[i]] = 1 - W/dist[x[i],y[i]]  
            KappaL = KappaU

    KappaU = KappaU + tp(KappaU)
    KappaL = KappaL + tp(KappaL)
    KappaU[np.abs(KappaU) < eps] = 0
    KappaL[np.abs(KappaL) < eps] = 0
    
    return KappaL, KappaU


#--------------------------Geodesic distance matrix
def distGeo(adjacency_matrix):
    '''All pair shortest path using Floyd-Warshall algorithm
    Input
        An NxN NumPy array describing the directed distances between N nodes.
        adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)
        Notes:
        * If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
        * The diagonal of adjacency_matrix should be zero.
    Output
        An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
    '''
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 

    return mat     

def check_and_convert_adjacency_matrix(adjacency_matrix):
    mat = asarray(adjacency_matrix)

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows
    
    #change zero elements to inf and zero diagonals
    adjacency_matrix[adjacency_matrix==0] = inf
    np.fill_diagonal(adjacency_matrix, 0)
    
    assert (diagonal(mat) == 0.0).all()

    return (mat, n)

#--------------------------Diffusion distance matrix
def distDiff(A, t, l):

    N = A.shape[0]
    eps = np.finfo(float).eps
    D = diag(np.sum(A, 1))#degree matrix

    # L = D\A 
    # L = D^(-1/2)*A*D^(-1/2) #random walk Laplacian
    L = D - A #combinatorial Laplacian
    L = matprod([fracpow(D, -1/2), L, fracpow(D, -1/2)])
    
    
    # compute diffusion by matrix exponential
    Phi = expm(-t*L)
    Phi[Phi < eps] = 0
    
    # compute diffusion by eigenvalue decomposition
    # [V,lamb] = eig(L) #s(L,l,'la')
    # lamb = diag(lamb)
    # I = eye(N) #initial condition for each vertex
    # Phi0V = tp(V)*I
    # Phi = tp(V).*exp(-lamb*t) #diffusion map
    # Phi = V*Phi
    
    # all diffusion distances 
    d = zeros([N, N])
    for i in range(N):
        for j in range(i-1,N):
            d[i,j] = norm( Phi[:,i] - Phi[:,j] )

    d = d + tp(d)
    d[d<eps] = 0
    
    return d

#--------------------------Diffusion 
def Diff(A, t, l):
    #The distance is measured on the graph G=(V,E,w), which is a Matlab object.

    eps = np.finfo(float).eps
    D = diag(np.sum(A, 1))#degree matrix

    # L = D\A 
    # L = D^(-1/2)*A*D^(-1/2) #random walk Laplacian
    L = D - A #combinatorial Laplacian
    L = matprod([fracpow(D, -1/2), L, fracpow(D, -1/2)])
    
    
    # compute diffusion by matrix exponential
    Phi = expm(-t*L)
    Phi[Phi < eps] = 0
    
    # compute diffusion by eigenvalue decomposition
    # [V,lamb] = eig(L) #s(L,l,'la')
    # lamb = diag(lamb)
    # I = eye(N) #initial condition for each vertex
    # Phi0V = tp(V)*I
    # Phi = tp(V).*exp(-lamb*t) #diffusion map
    # Phi = V*Phi
    
    return Phi
                 
          
#--------------------------Exact optimal transport problem (lin program)  
''' Wasserstein distance (Hitchcock optimal transportation problem) between
 measures mx, my.
 beq is 1 x (m+n) vector [mx,my] of the one-step probability distributions
 mx and my at x and y, respectively
 d is 1 x m*n vector of distances between supp(mx) and supp(my)    
'''
def W1(mx,my,dist):

    nmx = len(mx) 
    nmy = len(my)
    dist = np.reshape(tp(dist), nmx*nmy)

    A = cat( (kron(ones([1,nmy], dtype=int),eye(nmx, dtype=int)), kron(eye(nmy, dtype=int),ones([1,nmx], dtype=int))),axis=0 )
    beq = cat((mx, my),axis=0)

    #options = optimoptions('linprog','Algorithm','dual-simplex','display','off')
    # maxIt = 1e5 
    #tol = 1e-9
    fval = linprog(dist, A_eq=A, b_eq=beq, method='simplex')
       
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
    
def sinkhornTransport(mx,my,K,U,lamb,stoppingCriterion = 'marginalDifference',p_norm=inf,tolerance=0.005,maxIter=5000,VERBOSE=0):
  
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
        I=(mx>0)
        if np.all(I) == 0: # need to update some vectors and matrices if mx does not have full support
            K=K[I,:]
            U=U[I,:]
            mx=mx[I]

        ainvK=divide(K,mx) # precomputation of this matrix saves a d1 x N Schur product at each iteration.

    # Fixed point counter
    compt=0

    # Initialization of Left scaling Factors, N column vectors.
    uL=divide(ones([mx.shape[0],my.shape[1]]), mx.shape[0])

    if stoppingCriterion == 'distanceRelativeDecrease':
        Dold=ones([1,my.shape[1]]) #initialization of vector of distances.

    # Fixed Point Loop
    # The computation below is mostly captured by the repeated iteration of uL=mx./(K*(my./(K'*uL)))
    while compt<maxIter:
        if ONE_VS_N: # 1-vs-N mode
            if BIGN:
                uL=divide(1, ainvK@divide(my, tp(K)@uL)) # main iteration of Sinkhorn's algorithm
            else:
                uL=divide(1, ainvK@divide(my, tp(tp(uL)@csc_matrix(K))))
        else: # N times 1-vs-1 mode
            if BIGN:
                uL=divide(mx, K@divide(my, tp(tp(uL)@csc_matrix(K))))
            else:
                uL=divide(mx, K@divide(my, tp(csc_matrix(K))@uL))

        compt=compt+1
    
        # check the stopping criterion every 20 fixed point iterations
        # or, if that's the case, before the final iteration to store the most
        # recent value for the matrix of right scaling factors vR.
        if mod(compt,20)==1 or compt==maxIter:   
            # split computations to recover right and left scalings.        
            if BIGN:
                vR=divide(my, tp(csc_matrix(K))@uL) # main iteration of Sinkhorn's algorithm
            else:
                vR=divide(my, tp(tp(uL)@csc_matrix(K)))
        
            if ONE_VS_N: # 1-vs-N mode
                uL=divide(1, ainvK@vR)
            else:
                uL=divide(mx, csc_matrix(K)@vR)       
                        
            # check stopping criterion
            if stoppingCriterion == 'distanceRelativeDecrease':
                D=np.sum(multiply(uL, U@vR))
                Criterion=norm(divide(D, Dold-1,p_norm))
                if Criterion<tolerance or isnan(Criterion):
                    break                
                Dold=D               
            elif stoppingCriterion == 'marginalDifference':
                Criterion=norm(np.array(np.sum(np.abs( multiply(vR, tp(csc_matrix(K))@uL) - my )),ndmin=2),p_norm)
                if Criterion<tolerance or isnan(Criterion): # npl.norm of all or . or_1 differences between the marginal of the current solution with the actual marginals.
                    break
            else:
                sys.exit('Stopping Criterion not recognized')        
      
            compt=compt+1
            if VERBOSE>0:
               print('Iteration :',compt,' Criterion: ',Criterion)        
        
            if np.any(isnan(Criterion)): # stop all computation if a computation of one of the pairs goes wrong.
                sys.exit('NaN values have appeared during the fixed point iteration. This problem appears because of insufficient machine precision when processing computations with a regularization value of lamb that is too high. Try again with a reduced regularization parameter lamb or with a thresholded metric matrix M.')

    if stoppingCriterion == 'marginalDifference': # if we have been watching marginal differences, we need to compute the vector of distances.
        D=np.sum(multiply(uL, U@vR))

    alpha = log(uL)
    beta = log(vR)
    beta[beta==-inf]=0 # zero values of vR (corresponding to zero values in my) generate inf numbers.
    if ONE_VS_N:
        L = (tp(mx)@alpha + np.sum(multiply(my, beta)))/lamb
    else:       
        alpha[alpha==-inf]=0 # zero values of uL (corresponding to zero values in mx) generate inf numbers. in ONE-VS-ONE mode this never happens.
        L = (np.sum(multiply(mx, alpha)) + np.sum(multiply(my, beta)))/lamb
        
    return D, L

    
#--------------------------Plot
def plotCluster(G,T,N,comms,X,Y,f):
    import plotly.plotly as py
    import plotly.graph_objs as go
    import networkx as nx


    G=nx.random_geometric_graph(200,0.125)
    pos=nx.get_node_attributes(G,'pos')

    dmin=1
    ncenter=0
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d

    p=nx.single_source_shortest_path_length(G,ncenter)
    
    edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
        
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = '# of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])   
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    py.iplot(fig, filename='networkx')
#    # plot
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