import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from multiprocessing import Pool
from functools import partial
from scipy.sparse.csgraph import laplacian, floyd_warshall
import ot
import spectral_embedding_signed as ses


class Geometric_Clustering(object): 

    def __init__(self, G, T=np.logspace(0,1,10), laplacian_tpe='normalized',\
                 cutoff=1., lamb=0, GPU=False, workers=2):

        #set the graph
        self.G = G
        self.A = nx.adjacency_matrix(self.G, weight='weight')
        self.n = len(G.nodes)
        self.e = len(G.edges)
        self.laplacian_tpe = laplacian_tpe
        self.labels_gt = []
        if 'block' in G.node[0]:
            for i in self.G:
                self.labels_gt = np.append(self.labels_gt,self.G.node[i]['block'])

        #time vector
        self.n_t = len(T)
        self.T = T

        #precision parameters
        self.cutoff = cutoff
        self.lamb = lamb

        #GPU and cpu parameters
        self.GPU = GPU
        self.workers = workers


    def construct_laplacian(self): 
        """Laplacian matrix"""
        
        print('\nConstruct ' + self.laplacian_tpe + ' Laplacian')
        
        if self.laplacian_tpe == 'normalized':
            degree = np.array(self.A.sum(1)).flatten()
            self.L = sc.sparse.csr_matrix(nx.laplacian_matrix(self.G).toarray().dot(np.diag(1./degree))) 

        elif self.laplacian_tpe == 'combinatorial':
            self.L = 1.*laplacian(self.A, normed=False, return_diag=False, use_out_degree=False)


    def compute_distance_geodesic(self):
        """Geodesic distance matrix"""
        
        print('\nCompute geodesic distance matrix')

        self.dist = floyd_warshall(self.A, directed=True, unweighted=False)

       
    def compute_OR_curvatures(self):
        """Edge curvature matrix"""    
        
        print('\nGraph: ' + self.G.graph['name'])
        
        self.construct_laplacian() #Laplacian matrix 
        self.compute_distance_geodesic() #Geodesic distance matrix
        
        print('\nCompute measures')

        with Pool(processes = self.workers) as p_mx: 
            mx_all = list(tqdm(p_mx.imap(partial(mx_comp, self.L, self.T, self.cutoff), self.G.nodes()), total = self.n))
        
        if self.cutoff < 1. or self.lamb == 0:
            print('\nCompute edge curvatures')

            with Pool(processes = self.workers) as p_kappa:  
                Kappa = list(tqdm(p_kappa.imap(partial(K_ij, mx_all, self.dist, self.lamb), self.G.edges()), total = self.e))
            
            #curvature matrix of size (edges x time) 
            Kappa = np.transpose(np.stack(Kappa, axis=1))
            
        elif self.cutoff == 1. and self.lamb > 0:
            assert self.lamb > 0, 'Lambda must be greater than zero'
               
            print('\nCompute the edge curvatures')
            
            n = nx.number_of_nodes(self.G)
            e = nx.Graph.size(self.G)
            Kappa = np.empty((e,self.n_t))
            for it in range(self.n_t): 
                print('    ... at Markov time ' + str(self.T[it]))
                mx_all_t = np.empty([n,n]) 
                for i in range(len(mx_all)):
                    mx_all_t[:,[i]] = mx_all[i][0][it].toarray()
                
                if self.GPU:
                    Kappa[:,it] = K_all_gpu(mx_all_t,self.dist,self.lamb,self.G)  
                else:
                    Kappa[:,it] = K_all(mx_all_t,self.dist,self.lamb,self.G)  

        self.Kappa = Kappa


    def compute_node_curvature(self):
        """Node curvatures from the adjacent edge curvatures"""

        B = nx.incidence_matrix(self.G).toarray() #incidence matrix with only ones (no negative values)
        Dinv = np.diag(1./B.sum(1)) # inverse degree matrix

        self.Kappa_node = Dinv.dot(B).dot(self.Kappa)


    def run_clustering(self,cluster_tpe='threshold', cluster_by='curvature'):
        """Clustering of curvature weigthed graphs"""
        
        self.cluster_tpe = cluster_tpe
        self.n_t = len(self.T)

        if cluster_tpe == 'threshold':
            
            nComms = np.zeros(self.n_t) 
            MIs = np.zeros(self.n_t) 
            labels = np.zeros([self.n_t, self.n])

            for i in tqdm(range(self.n_t)):

                # update edge curvatures in G
                for e, edge in enumerate(self.G.edges):
                    self.G.edges[edge]['curvature'] = self.Kappa[e,i]                     

                nComms[i], MIs[i], labels[i] = cluster_threshold(self)
                
            self.clustering_results = {'Markov time' : self.T,
                    'number_of_communities' : nComms,
                    'community_id' : labels,
                    'MI' : MIs}    

        else:
            import pygenstability.pygenstability as pgs

            #parameters
            stability = pgs.PyGenStability(self.G.copy(), cluster_tpe, louvain_runs=10, precision=1e-6)
            stability.all_mi = False #to compute MI between all Louvain runs
            stability.n_mi = 20  #if all_mi = False, number of top Louvain run to use for MI        
            stability.n_processes_louv = 2 #number of cpus 
            stability.n_processes_mi = 2 #number of cpus

            stabilities = []; nComms = []; MIs = []; labels = []
            for i in tqdm(range(self.n_t)):

                #set adjacency matrix
                if cluster_by == 'curvature':                 
                    for e, edge in enumerate(self.G.edges):
                        stability.G.edges[edge]['curvature'] = self.Kappa[e,i] 
                    stability.A = nx.adjacency_matrix(stability.G, weight='curvature')
                elif cluster_by == 'weight' :   
                    stability.A = nx.adjacency_matrix(stability.G, weight='weight')

                #run stability and collect results
                stability.run_single_stability(time = 1.)
                stabilities.append(stability.single_stability_result['stability'])
                nComms.append(stability.single_stability_result['number_of_comms'])
                MIs.append(stability.single_stability_result['MI'])
                labels.append(stability.single_stability_result['community_id'])

            ttprime = stability.compute_ttprime(labels, nComms, self.T)

            #save the results
            self.clustering_results = {'Markov time' : self.T,
                    'stability' : stabilities,
                    'number_of_communities' : nComms,
                    'community_id' : labels,
                    'MI' : MIs, 
                    'ttprime': ttprime}

    def run_embedding(self):
        pos = nx.get_node_attributes(self.G, 'pos')
        xyz = []
        for i in range(len(pos)):
            xyz.append(pos[i])
        xyz = np.array(xyz)
            
        self.Y = []
        for k in range(self.Kappa.shape[1]):            
            A = np.zeros([self.n,self.n])
            for i,edge in enumerate(self.G.edges):
                A[edge] = self.Kappa[i,k]
                A[edge[::-1]] = self.Kappa[i,k]
                
            se = ses.SpectralEmbedding(n_components=2,affinity='precomputed')
            self.Y.append(se.fit_transform(A))


'''
=============================================================================
Functions for computing the curvature
=============================================================================
'''

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
def K_ij(mx_all, dist, lamb, e):
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


def K_all(mx_all, dist, lamb, G):     
       
    dist = dist.astype(float)
    
    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in tqdm(x):
        ind = [y[1] for y in G.edges if y[0] == i]              

        W = ot.sinkhorn(mx_all[:,i].tolist(), mx_all[:,ind].tolist(), dist.tolist(), lamb)    
        Kt = np.append(Kt, 1. - W/dist[i][ind])
        
    return Kt


def K_all_gpu(mx_all, dist, lamb, G):   
    import ot.gpu    
       
    mx_all = ot.gpu.to_gpu(mx_all) 
    dist = ot.gpu.to_gpu(dist.astype(float))
    lamb = ot.gpu.to_gpu(lamb)

    dist = dist.astype(float)
    
    Kt = []
    x = np.unique([x[0] for x in G.edges])
    for i in x:
        ind = [y[1] for y in G.edges if y[0] == i]              

        W = ot.gpu.sinkhorn(mx_all[:,i].tolist(), mx_all[:,ind].tolist(), dist.tolist(), lamb)    
        Kt = np.append(Kt, 1. - W/ot.gpu.to_np(dist[i][ind]))
        
    return Kt


def cluster_threshold(self,  sample=20, perturb=0.02):
    #parameters
    #sample = 20     # how many samples to use for computing the VI
    #perturb = 0.02  # threshold k ~ Norm(0,perturb(kmax-kmin))

    Aold = self.A.toarray()
    K_tmp = nx.adjacency_matrix(self.G, weight='curvature').toarray()

    mink = np.min(K_tmp)
    maxk = np.max(K_tmp)
    labels = np.zeros([sample, Aold.shape[0] ])

    #set the first threshold to 0, others are random numbers around 0
    thres = np.append(0.0, np.random.normal(0, perturb*(maxk-mink), sample-1))

    nComms = np.zeros(sample)
    for k in range(sample):
        ind = np.where(K_tmp <= thres[k])     
        A = Aold.copy()
        A[ind[0], ind[1]] = 0 #remove edges with negative curvature.       
        nComms[k], labels[k] = sc.sparse.csgraph.connected_components(csr_matrix(A, dtype=int), directed=False, return_labels=True) 

    # compute the MI between the threshold=0 and other ones
    from sklearn.metrics.cluster import normalized_mutual_info_score

    mi = 0; k = 0 
    for i in range(sample):
            mi += normalized_mutual_info_score(list(labels[i]),list(self.labels_gt), average_method='arithmetic' )
            k+=1

    #return the mean number of communities, MI and label at threshold = 0 
    return np.mean(nComms), mi/k, labels[0]