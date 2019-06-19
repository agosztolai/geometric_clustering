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
import pandas as pd


class Geometric_Clustering(object):
    """
    Main class for geometric clustering
    """

    def __init__(self, G=[], pos=[], laplacian_tpe='normalized', \
                 t_min=0, t_max = 1, n_t = 100, log=True, cutoff=0.95, \
                 lamb=0, GPU=False, workers=2, node_labels=False, filename = 'res'):

        #set the graph
        self.G = G
        self.n = len(G.nodes)
        self.m = len(G.edges)
        
        self.filename = filename

        #time vector
        self.log = log
        if log:
            self.T = np.logspace(t_min, t_max, n_t)
        else:
            self.T = np.linspace(t_min, t_max, n_t)

        #precision parameters
        self.cutoff = cutoff
        self.lamb = lamb
    
        #cluster with threshold parameters
        self.sample = 20     # how many samples to use for computing the VI
        self.perturb = 0.02  # threshold k ~ Norm(0,perturb(kmax-kmin))

        #GPU and cpu parameters
        self.GPU = GPU
        self.workers = workers

        #plotting parameters
        self.figsize = None #(5,4)
        self.node_labels = node_labels

        #if no positions given, use force atlas
        if len(pos) == 0:
            forceatlas2 = ForceAtlas2(
                        scalingRatio=0.5,
                        strongGravityMode=False,
                        gravity=1.0,
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        verbose=False)

            self.pos = forceatlas2.forceatlas2_networkx_layout(self.G, pos=None, iterations=2000)
        else: #else use positions
            self.pos = pos

        #save the Laplacian matrix and adjacency matrix
        self.laplacian_tpe = laplacian_tpe
        self.construct_laplacian()


    def construct_laplacian(self): 
        """
        save the Laplacian matrix for later
        """

        #save the adjacency matrix (sparse)
        self.A = nx.adjacency_matrix(self.G)

        #save Laplacian matrix
        if self.laplacian_tpe == 'normalized':
            degree = np.array(self.A.sum(1)).flatten()
            self.L = sc.sparse.csr_matrix(nx.laplacian_matrix(self.G).toarray().dot(np.diag(1./degree))) 
            #self.L = sc.sparse.csr_matrix((np.diag(1./degree)).dot(nx.laplacian_matrix(self.G).toarray()))

        elif self.laplacian_tpe == 'combinatorial':
            self.L = sc.sparse.csr_matrix(1.*nx.laplacian_matrix(self.G)) 


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

       
    def compute_OR_curvatures(self, disp = True):
        """
        # =============================================================================
        # Compute the Curvature matrix (parallelised)
        # =============================================================================
        """
      
        if disp:
            disable = False
        else:
            disable = True
        
        if not self.GPU:
            
            print('Compute the measures')

            with Pool(processes = self.workers) as p_mx:  #initialise the parallel computation
                mx_all = list(tqdm(p_mx.imap(partial(mx_comp, self.L, self.T, self.cutoff), self.G.nodes()), total = self.n, disable = disable))

            print('Compute the edge curvatures')

            with Pool(processes = self.workers) as p_kappa:  #initialise the parallel computation
                Kappa = list(tqdm(p_kappa.imap(partial(K_comp, mx_all, self.dist, self.lamb), self.G.edges()), total = self.m, disable = disable))   
            
            #curvature matrix of size (edges x time) 
            Kappa = np.transpose(np.stack(Kappa, axis=1))
            
        elif self.GPU:
            print("Not working yet!!!!!")
            cutoff = 1. #this is to keep dist matrix same, remove later if possible
            
            print('Compute the measures')
            
            with Pool(processes = self.workers) as p_mx:  #initialise the parallel computation
                mx_all = list(tqdm(p_mx.imap(partial(mx_comp, self.L, self.T, cutoff), self.G.nodes()), total = self.n, disable = disable))
            
            print('Compute the edge curvatures')
            
            n = nx.number_of_nodes(self.G)
            e = nx.Graph.size(self.G)
            Kappa = np.empty((e,len(self.T)))
            for it in tqdm(range(len(self.T))):           
                mx_all_t = np.empty([n,n]) 
                for i in range(len(mx_all)):
                    mx_all_t[:,[i]] = mx_all[i][0][it].toarray()
                
                Kappa[:,it] = K_comp_gpu(self.G,self.T,mx_all_t,self.dist,self.lamb)            

        self.Kappa = Kappa

    def node_curvature(self):
        """
        Compute node curvatures from the adjacent edge curvatures
        """

        B = nx.incidence_matrix(self.G).toarray() #incidence matrix with only ones (no negative values)
        Dinv = np.diag(1./B.sum(1)) # inverse degree matrix

        self.Kappa_node = Dinv.dot(B).dot(self.Kappa)


    def compute_ricci_flow(self, tau):
        """
        Compute the Ricci flow using forward Euler scheme
        """

        self.T_ricci = self.T.copy()
        self.T = [tau]
        dt = self.T_ricci[1] - self.T_ricci[0]

        def new_kappa(): 

            self.construct_laplacian()
            self.compute_distance_geodesic()
            self.compute_OR_curvatures(disp = False)

            return self.Kappa.flatten()

        def f(weight):

            for k, e in enumerate(self.G.edges):
                self.G[e[0]][e[1]]['weight'] = weight[k]

            self.kappa = new_kappa()

            kappa_mean = (self.kappa*weight).sum()/weight.sum()

            return -(self.kappa - kappa_mean)*weight
 
        weights = np.zeros([len(self.T_ricci)+1, self.m])
        weights[0] = np.array([self.G[i][j]['weight'] for i,j in self.G.edges])
        kappas = np.zeros([len(self.T_ricci), self.m])

        for i in tqdm(range(len(self.T_ricci))):

            weights[i+1] = weights[i] + dt*f(weights[i])
            kappas[i] = self.kappa

        self.Kappa = kappas.T
        self.Weights = weights[:-1].T

        return weights, kappas

# =============================================================================
# Clustering functions
# =============================================================================

    def cluster_threshold(self):
        """
        Find threshold cluster using weights kappa in graph self.G 
        """

        Aold = nx.adjacency_matrix(self.G).toarray()
        K_tmp = nx.adjacency_matrix(self.G, weight='kappa').toarray()

        mink = np.min(K_tmp)
        maxk = np.max(K_tmp)
        labels = np.zeros([self.sample, Aold.shape[0] ])

        #set the first threshold to 0, others are random numbers around 0
        thres = np.append(0.0, np.random.normal(0, self.perturb*(maxk-mink), self.sample-1))

        nComms = np.zeros(self.sample)
        for k in range(self.sample):
            ind = np.where(K_tmp <= thres[k])     
            A = Aold.copy()
            A[ind[0], ind[1]] = 0 #remove edges with negative curvature.       
            nComms[k], labels[k] = sc.sparse.csgraph.connected_components(csr_matrix(A, dtype=int), directed=False, return_labels=True) 

        # compute the MI between the threshold=0 and other ones
        from sklearn.metrics.cluster import normalized_mutual_info_score

        mi = 0
        k = 0 
        for i in range(self.sample):
            for j in range(i-1):
                j=0
                mi += normalized_mutual_info_score(list(labels[i]),list(labels[j]), average_method='arithmetic' )
                k+=1

        #return the mean number of communities, MI and label at threshold = 0 
        return np.mean(nComms), mi/k, labels[0]


    def clustering(self):
        """
        Apply signed clustering on the curvature weigthed graphs
        """

        # cluster
        if self.cluster_tpe == 'threshold':
            nComms = np.zeros(len(self.T)) 
            MIs = np.zeros(len(self.T)) 
            labels = np.zeros([len(self.T), self.n]) 

            for i in tqdm(range(len((self.T)))):

                # update edge curvatures in G
                for e, edge in enumerate(self.G.edges):
                    self.G.edges[edge]['kappa'] = self.Kappa[e,i]                     

                nComms[i], MIs[i], labels[i] = self.cluster_threshold()

        if self.cluster_tpe == 'modularity':

            import PyGenStability as pgs
            
            louvain_runs = 10
            precision = 1e-6

            G_modularity = self.G.copy()

            self.stability = pgs.PyGenStability(G_modularity, 'modularity_signed', louvain_runs , precision)
            self.stability.all_mi = False #to compute MI between al Louvain
            self.stability.n_mi = 10  #if all_mi = False, number of top Louvai run to use for MI        
            self.stability.n_processes_louv = 2 #number of cpus 
            self.stability.n_processes_mi = 2 #number of cpus

            stabilities = []
            nComms = []
            MIs = []
            labels = []
            for i in tqdm(range(len((self.T)))):

                # update edge weights in stability.G
                for e, edge in enumerate(self.G.edges):
                    self.stability.G.edges[edge]['weight'] = self.Kappa[e,i]            

                self.stability.A = nx.adjacency_matrix(self.stability.G, weight='weight')
                self.stability.run_single_stability(time = 1.)

                stabilities.append(self.stability.single_stability_result['stability'])
                nComms.append(self.stability.single_stability_result['number_of_comms'])
                MIs.append(self.stability.single_stability_result['MI'])
                labels.append(self.stability.single_stability_result['community_id'])

            ttprime = self.stability.compute_ttprime(labels, nComms, self.T)

            #save the results
            timesteps = [element[0] for element in enumerate(self.T)]
            self.stability.stability_results = pd.DataFrame(
                {
                    'Markov time' : self.T,
                    'stability' : stabilities,
                    'number_of_communities' : nComms,
                    'community_id' : labels,
                    'MI' : MIs, 
                    'ttprime': ttprime
                },
                index = timesteps,
            )
                            
        self.nComms = nComms
        self.MIs = MIs
        self.labels = labels

# =============================================================================
# save/load functions
# =============================================================================

    def save_curvature(self):
        pickle.dump(self.Kappa, open(self.filename + '.pkl','wb'))

    def save_ricci_flow(self):
        pickle.dump([self.Kappa, self.Weights], open(self.filename + '.pkl','wb'))

    def save_clustering(self):

        if self.cluster_tpe == 'threshold':
            pickle.dump([self.G, self.nComms, self.MIs, self.labels], open(self.filename + '.pkl','wb'))
        else:
            pickle.dump([self.G, self.stability.stability_results], open(self.filename + '.pkl','wb'))

    def load_curvature(self):
        self.Kappa = pickle.load(open(self.filename + '.pkl','rb'))

    def load_ricci_flow(self):
        self.Kappa, self.Weights = pickle.load(open(self.filename + '.pkl','rb'))

    def load_clustering(self):

        if self.cluster_tpe == 'threshold':
            self.G, self.nComms, self.MIs, self.labels = pickle.load(open(self.filename + '.pkl','rb'))
        else:           
            louvain_runs = 10
            precision = 1e-6

            G_modularity = self.G.copy()

            import PyGenStability as pgs

            self.stability = pgs.PyGenStability(G_modularity, 'modularity_signed', louvain_runs, precision)
            self.stability.all_mi = False #to compute MI between al Louvain
            self.stability.n_mi = 10  #if all_mi = False, number of top Louvain run to use for MI
            self.stability.n_processes_louv = 2 #number of cpus 
            self.stability.n_processes_mi = 2 #number of cpus 

            self.stability.stability_results = pickle.load(open(self.filename + '.pkl','rb'))
            self.labels = self.stability.stability_results['community_id']

# =============================================================================
# plotting functions
# =============================================================================

    def plot_curvature_graph(self, t, node_size  = 100, edge_width = 2):
        """
        plot the curvature on the graph for a given time t
        """

        plt.figure(figsize = self.figsize)

        edge_vmin = -np.max(abs(self.Kappa[:,t]))
        edge_vmax = np.max(abs(self.Kappa[:,t]))
        
        self.node_curvature()

        vmin = -np.max(abs(self.Kappa_node[:,t]))
        vmax = np.max(abs(self.Kappa_node[:,t]))

        nodes = nx.draw_networkx_nodes(self.G, pos = self.pos, node_size = node_size, node_color = self.Kappa_node[:,t], vmin = vmin, vmax = vmax,  cmap=plt.get_cmap('coolwarm'))
        edges = nx.draw_networkx_edges(self.G, pos = self.pos, width = edge_width, edge_color = self.Kappa[:, t], edge_vmin = edge_vmin, edge_vmax = edge_vmax, edge_cmap=plt.get_cmap('coolwarm'))

        plt.colorbar(edges, label='Edge curvature')

        if self.node_labels:
            old_labels={}
            for i in self.G:
                old_labels[i] = str(i) + ' ' + self.G.node[i]['old_label']
            nx.draw_networkx_labels(self.G, pos = self.pos, labels = old_labels)

        limits = plt.axis('off') #turn axis odd


    def video_curvature(self, n_plot = 10, folder = 'images_curvature', node_size = 100):
        """
        plot the curvature on the graph for each time
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

            plt.savefig(folder + '/curvature_' + str(i) + '.svg', bbox_inches='tight')
            plt.close()

    def plot_curvatures(self):
        """
        plot the curvature of each edge as a function of time
        """
        
        #plot the edge curvatures
        plt.figure(figsize = self.figsize)

        plt.plot(self.T, self.Kappa.mean(0), lw=5,c='C1',label='mean curvature')
        plt.fill_between(self.T,self.Kappa.mean(0)-self.Kappa.std(0),self.Kappa.mean(0)+self.Kappa.std(0), alpha=0.5, color='C0',label='std curvature')

        plt.axvline(self.T[np.argmax(self.Kappa.std(0))], ls='--', c='C0', label='max(std)')
        plt.axhline(np.min(self.Kappa.mean(0)),c='C1', ls='--', label='min(mean)')
        plt.axhline(0,ls='--',c='0.5',label='flat')

        plt.plot(self.T, self.Kappa[0], c='0.3', lw=0.2,label='edge curvature')
        for k in self.Kappa:
            plt.plot(self.T, k, c='0.4', lw=0.2)

        ax = plt.gca()
        if self.log:
            ax.set_xscale('log')

        plt.xlabel('Time')
        plt.ylabel('Edge curvature')
        plt.axis([self.T[0],self.T[-1],np.min(self.Kappa)-0.1,1 ])
        plt.legend(loc='best')
        plt.savefig('curvatures_edges.svg', bbox_inches = 'tight')

        #plot the node curvatures
        self.node_curvature()

        plt.figure(figsize = self.figsize)

        plt.plot(self.T, self.Kappa_node.mean(0), lw=5,c='C1', label='mean curvature')
        plt.fill_between(self.T,self.Kappa_node.mean(0)-self.Kappa_node.std(0),self.Kappa_node.mean(0)+self.Kappa_node.std(0), alpha=0.5,color='C0', label='std curvature')
        plt.axvline(self.T[np.argmax(self.Kappa_node.std(0))], ls='--', c='C0', label='max(std)')
        plt.axhline(np.min(self.Kappa.mean(0)),c='C1',ls='--', label='min(mean)')
        plt.axhline(0,ls='--',c='0.5', label='flat')

        plt.plot(self.T, self.Kappa_node[0], c='0.3', lw=0.2,label='node curvature')
        for k in self.Kappa_node:
            plt.plot(self.T, k, c='0.3', lw=0.2)

        ax = plt.gca()
        if self.log:
            ax.set_xscale('log')
        plt.xlabel('Time')
        plt.ylabel('Node curvature')
        plt.axis([self.T[0],self.T[-1], np.min(self.Kappa_node), np.max(self.Kappa_node) ])
        plt.legend(loc='best')
        plt.savefig('curvatures_nodes.svg', bbox_inches='tight')

    def plot_ricci_flow_graph(self, t, node_size  = 100, edge_width = 2):

        """
        plot the curvature on the graph for a given time t
        """

        plt.figure(figsize = self.figsize)

        edge_vmin = np.min(abs(self.Weights[:,t]))
        edge_vmax = np.max(abs(self.Weights[:,t]))
        
        nodes = nx.draw_networkx_nodes(self.G, pos = self.pos, node_size = node_size) #, node_color = self.Kappa_node[:,t], vmin = vmin, vmax = vmax,  cmap=plt.get_cmap('coolwarm'))
        edges = nx.draw_networkx_edges(self.G, pos = self.pos, width = edge_width, edge_color = self.Weights[:, t], edge_vmin = edge_vmin, edge_vmax = edge_vmax, edge_cmap=plt.get_cmap('plasma'))

        plt.colorbar(edges, label='Edge weight')

        if self.node_labels:
            old_labels={}
            for i in self.G:
                old_labels[i] = str(i) + ' ' + self.G.node[i]['old_label']
            nx.draw_networkx_labels(self.G, pos = self.pos, labels = old_labels)

        limits = plt.axis('off') #turn axis odd


    def video_ricci_flow(self, n_plot = 10, folder = 'images_ricci_flow', node_size = 100):
        """
        plot the curvature on the graph for each time
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

            self.plot_ricci_flow_graph(t, node_size = node_size)

            plt.title(r'$log_{10}(t)=$'+str(np.around(np.log10(self.T[t]),2)))

            plt.savefig(folder + '/weight_' + str(i) + '.svg', bbox_inches='tight')
            plt.close()


    def plot_ricci_flow(self):
        """
        plot the curvature of each edge as a function of time
        """

        #plot the edge curvatures
        plt.figure(figsize = self.figsize)

        plt.plot(self.T, self.Weights.mean(0), lw=5,c='C1',label='mean weight')
        plt.fill_between(self.T,self.Weights.mean(0)-self.Weights.std(0),self.Weights.mean(0)+self.Weights.std(0), alpha=0.5, color='C0',label='std weight')

        plt.axvline(self.T[np.argmax(self.Weights.std(0))], ls='--', c='C0', label='max(std)')
        plt.axhline(np.min(self.Weights.mean(0)),c='C1', ls='--', label='min(mean)')
        plt.axhline(0,ls='--',c='0.5',label='flat')

        plt.plot(self.T, self.Weights[0], c='0.3', lw=0.2,label='edge curvature')
        for k in self.Weights:
            plt.plot(self.T, k, c='0.3', lw=0.2)

        ax = plt.gca()
        if self.log:
            ax.set_xscale('log')

        plt.xlabel('Time')
        plt.ylabel('Edge weight')
        plt.axis([self.T[0],self.T[-1],np.min(self.Weights),np.max(self.Weights) ])
        plt.legend(loc='best')
        plt.savefig('weights_edges.svg', bbox_inches = 'tight')


    def plot_clustering(self):
        """
        plot the clustering results
        """

        if self.cluster_tpe == 'modularity':
            self.stability.plot_scan(time_axis=True) #if time_axis=True, the ttprime is wrong (bug to fix)
        else:
            plt.figure(figsize=self.figsize)
            ax1 = plt.gca()
            ax1.semilogx(self.T, self.nComms, 'C0')

            ax1.set_xlabel('Markov time')
            ax1.set_ylabel('# communities', color='C0')

            ax2 = ax1.twinx()
            ax2.semilogx(self.T, self.MIs, 'C1')

            ax2.set_ylabel('Average mutual information', color='C1')

        plt.savefig('clustering.svg', bbox_inches = 'tight')

 
    def plot_clustering_graph(self, t, node_size  = 100, edge_width = 2):

        """
        plot the curvature on the graph for a given time t
        """

        plt.figure(figsize = self.figsize)

        edge_vmin = -np.max(abs(self.Kappa[:,t]))
        edge_vmax = np.max(abs(self.Kappa[:,t]))
        
        nodes = nx.draw_networkx_nodes(self.G, pos = self.pos, node_size = node_size, node_color = self.labels[t], cmap=plt.get_cmap("tab20"))
        edges = nx.draw_networkx_edges(self.G, pos = self.pos, width = edge_width, edge_color = self.Kappa[:, t], edge_vmin = edge_vmin, edge_vmax = edge_vmax, edge_cmap=plt.get_cmap('coolwarm'))

        plt.colorbar(edges, label='Edge curvature')

        if self.node_labels:
            old_labels={}
            for i in self.G:
                old_labels[i] = str(i) + ' ' + self.G.node[i]['old_label']
            nx.draw_networkx_labels(self.G, pos = self.pos, labels = old_labels)


        limits = plt.axis('off') #turn axis odd


    def video_clustering(self, n_plot = 10, folder = 'images_clustering', node_size = 100):
        """
        plot the curvature on the graph for each time
        """

        #create folder it not already there
        if not os.path.isdir(folder):
            os.mkdir(folder)

        print('plot clustering images')
        if n_plot > len(self.T)-1:
            n_plot = len(self.T)-1

        dt = int(len(self.T)/n_plot)
        for i in tqdm(range(n_plot)):
            t = i*dt   

            self.plot_clustering_graph(t, node_size = node_size)

            plt.title(r'$log_{10}(t)=$'+str(np.around(np.log10(self.T[t]),2)))

            plt.savefig(folder + '/clustering_' + str(i) + '.svg', bbox_inches='tight')
            plt.close()

# =============================================================================
# functions for parallel computations
# =============================================================================

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

        if lamb != 0: #entropy regularized OT
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