import numpy as np
import networkx as nx
import pylab as plt
import matplotlib as mpl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as conncomp

from funs import ORcurvAll_sparse_full, distGeo, Diff, plotCluster


# parameters
t = np.logspace(-1, 1, 20)  # diffusion time scale
l = 4                    # num. retained evals (currently not relevant)
prec = 0.99              # fraction of mass to retain
lamb = 0.               # regularising parameter (the larger the more accurate, but higher cost, and too large can blow up)
vis = 0                  # plot visible (does not work yet)

# load graph
#if not exist('G','var'):
 #   [G,A,X,Y] = inputGraphs(13) #graph
A = np.array([[ 0., 24, 24, 76, 26],
           [62,  0, 88, 46, 73],
           [79,  7,  0, 29, 55],
           [63, 95,  8,  0, 14],
           [93,  5,  4, 58,  0]])

#G = nx.Graph(A)

N = 20
G = nx.newman_watts_strogatz_graph(N, 2, 0.20)


#pos = nx.spring_layout(G)
x = np.linspace(0,2*np.pi,N)
posx = np.cos(x)
posy = np.sin(x)
pos= []
for i in range(N):
    pos.append([posx[i],posy[i]])


A = nx.to_numpy_matrix(G) 
L = nx.normalized_laplacian_matrix(G)

#plt.show()

# geodesic distances OK
dist = np.array(distGeo(A))

numcomms = np.zeros(len(t)) #v = 0 
for i in range(len(t)):
    print(' Diffusion time: ', t[i])

    # compute diffusion after time t[i] OK
    Phi = Diff(L,t[i],l)

    # compute curvatures
    (KappaL,KappaU) = ORcurvAll_sparse_full(A,dist,Phi,prec,lamb)

    c = np.array([KappaL[i][j] for i,j in G.edges])

    plt.figure()


    nx.draw(G, pos = pos, node_size=20, node_color='k')
    print(np.min(c))
    print(np.max(c))
    edges = nx.draw_networkx_edges(G, pos = pos, edge_color=c, width = 2, edge_cmap = plt.cm.bwr, edge_vmin = -np.max(abs(c)), edge_vmax = np.max(abs(c)))

    plt.savefig('images/t_'+str(i)+'.png')

    plt.close() 
    plt.figure()
    plt.hist(c,bins=50)
    plt.xlabel('curvature')
    plt.savefig('images/hist_'+str(i)+'.png')

    # update edge weights and curvatures
    #G.Edges.Weight = nonzero(tril(A))
    #indnonzeros = where(tril(A)>0) #edges with positive weights may have 0 kappa
    #G.Edges.Kappa = K(indnonzeros)
    
    # cluster  
    #ind = where(G.Edges.Kappa <= 0)
    #G1 = rmedge(G,ind) #remove edges with small curvature
    #comms = conncomp(G1)
    #N[i] = max(comms)

    Aclust = A.copy()
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
