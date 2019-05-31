import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from multiprocessing import Pool
from functools import partial
import ot
import os as os
import pickle as pickle
from fa2 import ForceAtlas2
import pylab as plt




class Geometric_Clustering(object):
    """
    Main class for geometric clustering
    """

    def __init__(self, G = [], pos = [], laplacian_tpe = 'normalized', T =[0,1], cutoff = 0.95,  lamb = 0, GPU = False, workers = 2, node_labels = False):

        #set the graph
        self.G = G
        self.n = len(G.nodes)
        self.m = len(G.edges)

        #time vector
        self.T = T

        #precision parameters
        self.cutoff = cutoff
        self.lamb = 0
    
        #GPU and cpu parameters
        self.GPU = GPU
        self.workers = workers

        #plotting parameters
        self.figsize = None #(5,4)
        self.labels = node_labels


        #if no positions given, use force atlas
        if len(pos) == 0:
            forceatlas2 = ForceAtlas2(
                        # Tuning
                        scalingRatio=2.,
                        strongGravityMode=False,
                        gravity=1.0,
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        # Log
                        verbose=False)

            self.pos = forceatlas2.forceatlas2_networkx_layout(self.G, pos=None, iterations=2000)
        else: #else use positions
            self.pos = pos


        #save the adjacency matrix (sparse)
        self.A = nx.adjacency_matrix(self.G)
        
        #save Laplacian matrix
        self.laplacian_tpe = laplacian_tpe
        if self.laplacian_tpe == 'normalized':
            self.L = sc.sparse.csc_matrix(nx.normalized_laplacian_matrix(G), dtype=np.float64)

        elif self.laplacian_tpe == 'combinatorial':
            self.L = sc.sparse.csr_matrix(1.*nx.laplacian_matrix(self.G)) #combinatorial Laplacian


    def compute_distance_geodesic(self):
        """
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
        """

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


        dist, n = check_and_convert_A(self.A.toarray())

        for k in range(n):
            dist = np.minimum(dist, dist[np.newaxis,k,:] + dist[:,k,np.newaxis]) 

        #save the distance matrix
        self.dist = dist

       
    def compute_OR_curvatures(self):
        """
        # =============================================================================
        # Compute the Curvature matrix (parallelised)
        # =============================================================================
        """

        
        if not self.GPU:
            print('Compute the measures')
            with Pool(processes = self.workers) as p_mx:  #initialise the parallel computation
                mx_all = list(tqdm(p_mx.imap(partial(mx_comp, self.L, self.T, self.cutoff), self.G.nodes()), total = self.n))

            print('Compute the edge curvatures')
            with Pool(processes = self.workers) as p_kappa:  #initialise the parallel computation
                Kappa = list(tqdm(p_kappa.imap(partial(K_comp, mx_all, self.dist, self.lamb), self.G.edges()), total = self.m))   
            #curvature of size (edges, time) 
            Kappa = np.transpose(np.stack(Kappa, axis=1))
            
        elif self.GPU:
            print("Not working yet!!!!!")
            cutoff = 1. #this is to keep distance matrix same, remove later if possible
            with Pool(processes = workers) as p_mx:  #initialise the parallel computation
                mx_all = list(tqdm(p_mx.imap(partial(mx_comp, L, T, cutoff), G.nodes()),\
                               total = len(G)))
            
            n = nx.number_of_nodes(G)
            e = nx.Graph.size(G)
            Kappa = np.empty((e,len(T)))
            for it in tqdm(range(len(T))):           
                mx_all_t = np.empty([n,n]) 
                for i in range(len(mx_all)):
                    mx_all_t[:,[i]] = mx_all[i][0][it].toarray()
                
                Kappa[:,it] = K_comp_gpu(G,T,mx_all_t,dist,lamb)    
           

        self.Kappa = Kappa

    def node_curvature(self):
        """
        compute the node curvature from the adjacent edge curvatures
        """

        B = nx.incidence_matrix(self.G).toarray() #incidence matrix with only ones (no negative values)
        Dinv = np.diag(1./B.sum(1)) # degree matrix

        self.Kappa_node = Dinv.dot(B).dot(self.Kappa)

    ##########################
    ## save/load functions ###
    ##########################

    def save_curvature(self):
        pickle.dump(self.Kappa, open('OR_results.pkl','wb'))

    def load_curvature(self):
        self.Kappa = pickle.load(open('OR_results.pkl','rb'))


    ########################
    ## plotting functions ##
    ########################

    def plot_curvature_graph(self, t, node_size  = 100, edge_width = 2):

        plt.figure(figsize = self.figsize)

        edge_vmin = -np.max(abs(self.Kappa[:,t]))
        edge_vmax = np.max(abs(self.Kappa[:,t]))
        
        self.node_curvature()

        vmin = -np.max(abs(self.Kappa_node[:,t]))
        vmax = np.max(abs(self.Kappa_node[:,t]))

        nodes = nx.draw_networkx_nodes(self.G, pos = self.pos, node_size = node_size, node_color = self.Kappa_node[:,t], vmin = vmin, vmax = vmax,  cmap=plt.get_cmap('coolwarm'))

        edges = nx.draw_networkx_edges(self.G, pos = self.pos, width = edge_width, edge_color = self.Kappa[:, t], edge_vmin = edge_vmin, edge_vmax = edge_vmax, edge_cmap=plt.get_cmap('coolwarm'))

        plt.colorbar(edges, label='Edge curvature')

        if self.labels:
            old_labels={}
            for i in self.G:
                old_labels[i] = str(i) + ' ' + self.G.node[i]['old_label']
            nx.draw_networkx_labels(self.G, pos = self.pos, labels = old_labels)


        limits = plt.axis('off') #turn axis odd




    def video_curvature(self, n_plot = 10, folder = 'images_curvature', node_size = 100):
        """
        plot the reachability for all tau
        """

        #create folder it not already there
        if not os.path.isdir(folder):
            os.mkdir(folder)

            print('plot curvature images')
        if n_plot > len(self.T)-1:
            n_plot = len(self.T)-1

        dt = int(len(self.T)/n_plot)
        for i in tqdm(range(n_plot)):
            t = i*dt   

            self.plot_curvature_graph(t, node_size = node_size)

            plt.title(r'$log_{10}(t)=$'+str(np.around(np.log10(self.T[t]),2)))

            plt.savefig(folder + '/curvature_' + str(i) + '.svg', bbox='tight')
            plt.close()

    def plot_curvatures(self):
        """
        plot the curvature of each edge as a function of time
        """

        #plot the edge curvatures
        plt.figure(figsize = self.figsize)

        plt.semilogx(self.T, self.Kappa.mean(0), lw=5,c='C1',label='mean curvature')
        plt.fill_between(self.T,self.Kappa.mean(0)-self.Kappa.std(0),self.Kappa.mean(0)+self.Kappa.std(0), alpha=0.5, color='C0',label='std curvature')

        plt.axvline(self.T[np.argmax(self.Kappa.std(0))], ls='--', c='C0', label='max(std)')
        plt.axhline(np.min(self.Kappa.mean(0)),c='C1', ls='--', label='min(mean)')
        plt.axhline(0,ls='--',c='0.5',label='flat')

        plt.semilogx(self.T, self.Kappa[0], c='0.3', lw=0.2,label='edge curvature')
        for k in self.Kappa:
            plt.semilogx(self.T, k, c='0.3', lw=0.2)


        plt.xlabel('Time')
        plt.ylabel('Edge curvature')
        plt.axis([self.T[0],self.T[-1],np.min(self.Kappa), 1])
        plt.legend(loc='best')
        plt.savefig('curvatures_edges.svg', bbox = 'tight')


        #plot the node curvatures
        self.node_curvature()

        plt.figure(figsize = self.figsize)

        plt.semilogx(self.T, self.Kappa_node.mean(0), lw=5,c='C1', label='mean curvature')
        plt.fill_between(self.T,self.Kappa_node.mean(0)-self.Kappa_node.std(0),self.Kappa_node.mean(0)+self.Kappa_node.std(0), alpha=0.5,color='C0', label='std curvature')
        plt.axvline(self.T[np.argmax(self.Kappa_node.std(0))], ls='--', c='C0', label='max(std)')
        plt.axhline(np.min(self.Kappa.mean(0)),c='C1',ls='--', label='min(mean)')
        plt.axhline(0,ls='--',c='0.5', label='flat')

        plt.semilogx(self.T, self.Kappa_node[0], c='0.3', lw=0.2,label='node curvature')
        for k in self.Kappa_node:
            plt.semilogx(self.T, k, c='0.3', lw=0.2)

        plt.xlabel('Time')
        plt.ylabel('Node curvature')
        plt.axis([self.T[0],self.T[-1],np.min(self.Kappa_node), 1])
        plt.legend(loc='best')
        plt.savefig('curvatures_nodes.svg', bbox='tight')


#########################################
## functions for parallel computations ##
#########################################

# compute all neighbourhood densities
def mx_comp(L, T, cutoff, i):
    N = np.shape(L)[0]

    def delta(i, n):
        p0 = np.zeros(n)
        p0[i] = 1.
        return p0


    mx_all = [] 
    Nx_all = []

    mx_tmp = delta(i, N) #set initial condition
    T = [0,] + list(T) #add time 0
    for i in range(len((T))-1): 
        #compute exponential by increments (faster than from 0)
        mx_tmp = sc.sparse.linalg.expm_multiply(-(T[i+1]-T[i])*L, mx_tmp)

        Nx = np.argwhere(mx_tmp >= (1-cutoff)*np.max(mx_tmp))
        mx_all.append(sc.sparse.lil_matrix(mx_tmp[Nx]/np.sum(mx_tmp[Nx])))
        Nx_all.append(Nx)

    return mx_all, Nx_all


# compute curvature for an edge ij
def K_comp(mx_all, dist, lamb, e):
    i = e[0]
    j = e[1]

    nt = len(mx_all[0][0])
    K = np.zeros(nt)
    for it in range(nt):

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


def K_comp_gpu(G, T, mx_all, dist, lamb): 
    import ot.gpu    
    
    mx_all = ot.gpu.to_gpu(mx_all) 
    dist = ot.gpu.to_gpu(dist.astype(float))
    lamb = ot.gpu.to_gpu(lamb)

    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in x:
        ind = [y[1] for y in G.edges if y[0] == i]              

        W = ot.gpu.sinkhorn(mx_all[:,i].tolist(), mx_all[:,ind].tolist(), dist.tolist(), lamb)    
        Kt = np.append(Kt, 1. - W/dist[i, ind])
        
    return Kt


