import numpy as np
import scipy as sc
from tqdm import tqdm
import networkx as nx
from multiprocessing import Pool
from functools import partial
from scipy.sparse.csgraph import laplacian, floyd_warshall
from sklearn.utils import check_symmetric
from .utils.curvature_utils import mx_comp, K_ij, K_all, K_all_gpu
from .utils.clustering_utils import cluster_threshold
from .utils.embedding_utils import signed_laplacian, SpectralEmbedding


class geocluster(object): 

    def __init__(self, G, T=np.logspace(0,1,10), laplacian_tpe='normalized',\
                 cutoff=0.99, lamb=0, GPU=False, workers=2, use_spectral_gap = True):

        #set the graph
        self.G = G
        self.A = check_symmetric(nx.adjacency_matrix(self.G, weight='weight'))
        self.n = len(G.nodes)
        self.e = len(G.edges)
        self.use_spectral_gap = use_spectral_gap
        self.laplacian_tpe = laplacian_tpe
        self.labels_gt = []
        if 'block' in G.nodes[0]:
            for i in self.G:
                self.labels_gt = np.append(self.labels_gt,self.G.nodes[i]['block'])

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

        elif self.laplacian_tpe == 'signed_normalized':
            self.L = signed_laplacian(self.A, normed=True, return_diag=True)

        if self.use_spectral_gap:
            self.L /= abs(sc.sparse.linalg.eigs(self.L, which='SM',k=2)[0][1])

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
               
            print('\nCompute the edge curvatures at all Markov times')
            
            n = nx.number_of_nodes(self.G)
            e = nx.Graph.size(self.G)
            Kappa = np.empty((e,self.n_t))
            for it in tqdm(range(self.n_t)): 
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

            for t in tqdm(range(self.n_t)):
                # update edge curvatures                    
                for e, edge in enumerate(self.G.edges):
                    self.G.edges[edge]['curvature'] = self.Kappa[e,t]                     

                nComms[t], MIs[t], labels[t] = cluster_threshold(self)
                
            self.clustering_results = {'Markov time' : self.T,
                    'number_of_communities' : nComms,
                    'community_id' : labels,
                    'MI' : MIs}    

        else:
            import pygenstability.pygenstability as pgs

            #parameters
            stability = pgs.PyGenStability(self.G.copy(), cluster_tpe, louvain_runs=50)
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
                    time = 1.
                elif cluster_by == 'weight' :   
                    stability.A = nx.adjacency_matrix(stability.G, weight='weight')
                    time = self.T[i]

                #run stability and collect results
                stability.run_single_stability(time = time ) 
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
        '''embedding based on curvature-signed Laplacian eigenmaps'''
        
#        pos = nx.get_node_attributes(self.G, 'pos')
#        xyz = []
#        for i in range(len(pos)):
#            xyz.append(pos[i])
#        xyz = np.array(xyz)
            
        self.Y = []
        for t in tqdm(range(self.n_t)):

            for e, edge in enumerate(self.G.edges):
                self.G.edges[edge]['curvature'] = self.Kappa[e,t]   
                
            se = SpectralEmbedding(n_components=2, affinity='precomputed')
            A = se.fit_transform(nx.adjacency_matrix(self.G, weight='curvature')).toarray()
            self.Y.append(A)
