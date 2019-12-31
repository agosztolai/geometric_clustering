#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import numpy as np
import scipy as sc
from tqdm import tqdm
import networkx as nx
import pickle as pickle
import pylab as plt

from multiprocessing import Pool
from functools import partial
from scipy.sparse.csgraph import laplacian, floyd_warshall
from sklearn.utils import check_symmetric

from .curvature import mx_comp, K_ij, K_all, K_all_gpu
from .clustering import cluster_threshold
from .embedding import signed_laplacian, SpectralEmbedding


class GeoCluster(object): 

    def __init__(self, G, T=np.logspace(0,1,10), laplacian_tpe='normalized',\
                 cutoff=1.0, lamb=0, GPU=False, workers=2, use_spectral_gap=True):

        #set the graph
        self.G = G
        self.A = check_symmetric(nx.adjacency_matrix(self.G, weight='weight'))
        self.n = len(G.nodes)
        self.e = len(G.edges)
        self.use_spectral_gap = use_spectral_gap
        self.laplacian_tpe = laplacian_tpe
#        if 'block' in G.nodes[0]:
        self.labels_gt = [int(self.G.nodes[i]['block']) for i in self.G.nodes if 'block' in self.G.nodes[i]]
        
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
               
            print('\nCompute the edge curvatures')
            
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


    def run_embedding(self, weight='curvature'):
        '''embedding based on curvature-signed Laplacian eigenmaps'''

        se = SpectralEmbedding(n_components=2, affinity='precomputed')
        A = se.fit_transform(nx.adjacency_matrix(self.G, weight=weight).toarray())

        return A 

    def run_embeddings(self, weight='curvature'):
        '''embedding based on curvature-signed Laplacian eigenmaps for all times'''
           
        self.Y = []
        for t in tqdm(range(self.n_t)):

            for e, edge in enumerate(self.G.edges):
                self.G.edges[edge]['curvature'] = self.Kappa[e,t]   
                
            self.Y.append(self.run_embedding(weight=weight))


    # =============================================================================
    # Functions for plotting
    # =============================================================================

    def plot_clustering(self, ext='.png'):
        """plot the clustering results"""
        
        import matplotlib.gridspec as gridspec
        plt.figure(figsize=(5,5))
        
        T = np.log10(self.clustering_results['Markov time'])
        
        if self.cluster_tpe == 'threshold':
            ax1 = plt.gca()
            ax1.semilogx(T, self.clustering_results['number_of_communities'], 'C0')
            ax1.set_xlabel('Markov time')
            ax1.set_ylabel('# communities', color='C0')

            ax2 = ax1.twinx()
            ax2.semilogx(T, self.clustering_results['MI'], 'C1')
            ax2.set_ylabel('Average mutual information', color='C1')
        else:
            #get the times paramters
            n_t = len(self.clustering_results['ttprime'])

            gs = gridspec.GridSpec(2, 1, height_ratios = [ 1., 0.5])
            gs.update(hspace=0)        

            #make the ttprime matrix
            ttprime = np.zeros([n_t,n_t])
            for i, tt in enumerate(self.clustering_results['ttprime']):
                ttprime[i] = tt 
                
            #plot tt'     
            ax0 = plt.subplot(gs[0, 0])
            ax0.contourf(T, T, ttprime, cmap='YlOrBr')
            ax0.yaxis.tick_left()
            ax0.yaxis.set_label_position('left')
            ax0.set_ylabel(r'$log_{10}(t^\prime)$')
            ax0.axis([T[0],T[-1],T[0],T[-1]])

            #plot the number of clusters
            ax1 = ax0.twinx()
            ax1.plot(T, self.clustering_results['number_of_communities'],c='C0',label='size',lw=2.)           
            ax1.yaxis.tick_right()
            ax1.tick_params('y', colors='C0')
            ax1.yaxis.set_label_position('right')
            ax1.set_ylabel('Number of clusters', color='C0')
        
            #plot the stability
            ax2 = plt.subplot(gs[1, 0])
            ax2.plot(T, self.clustering_results['stability'], label=r'$Q$',c='C2')
            #ax2.set_yscale('log') 
            ax2.tick_params('y', colors='C2')
            ax2.set_ylabel('Modularity', color='C2')
            ax2.yaxis.set_label_position('left')
            ax2.set_xlabel(r'$log_{10}(t)$')
            
            #plot the MMI
            ax3 = ax2.twinx()
            ax3.plot(T, self.clustering_results['MI'],'-',lw=2.,c='C3',label='MI')
            ax3.yaxis.tick_right()
            ax3.tick_params('y', colors='C3')
            ax3.set_ylabel(r'Mutual information', color='C3')
            ax3.axhline(1,ls='--',lw=1.,c='C3')
            ax3.axis([T[0], T[-1], 0,1.1])
            
        plt.savefig('clustering'+ext, bbox_inches = 'tight')
        
        
    def plot_graph(self, t, node_size=100, edge_width=2, node_labels=False, cluster=False):
        """plot the curvature on the graph for a given time t"""
        
            
        if 'pos' in self.G.nodes[0]:
            pos = list(nx.get_node_attributes(self.G,'pos').values())
        else:
            pos = nx.spring_layout(self.G)  
        
        if cluster:
            _labels = self.clustering_results['community_id'][t]
        else:
            _labels = [0] * self.n
            
        edge_vmin = -1. #-np.max(abs(self.Kappa[:,t]))
        edge_vmax = 1. #np.max(abs(self.Kappa[:,t]))    

        plt.figure(figsize = (5,4))
        if len(pos[0])>2:
            pos = np.asarray(pos)[:,[0,2]]

        nx.draw_networkx_nodes(self.G, pos=pos, node_size=node_size, node_color=_labels, cmap=plt.get_cmap("tab20"))
        nx.draw_networkx_edges(self.G, pos=pos, width=edge_width, edge_color=self.Kappa[:, t], edge_vmin=edge_vmin, edge_vmax=edge_vmax, edge_cmap=plt.cm.coolwarm, arrows=False)

        edges = plt.cm.ScalarMappable(
            norm = plt.cm.colors.Normalize(edge_vmin, edge_vmax),
          cmap = plt.cm.coolwarm)

        plt.colorbar(edges, label='Edge curvature')

        if node_labels:
            labels_gt={}
            for i in self.G:
                labels_gt[i] = str(i) + ' ' + str(self.G.nodes[i]['old_label'])
            nx.draw_networkx_labels(self.G, pos=pos, labels=labels_gt)

        plt.axis('off')


    def plot_edge_curvature(self, ext='.png'):

        plt.figure()
        plt.plot(np.log10(self.T), self.Kappa.T, c='C0', lw=0.5)
        plt.axvline(np.log10(self.T[np.argmax(np.std(self.Kappa.T,1))]), c='r', ls='--')
        plt.axhline(1, ls='--', c='k')
        plt.axhline(0, ls='--', c='k')
        plt.xlabel('log(time)')
        plt.ylabel('edge OR curvature')
        plt.savefig('edge_curvatures'+ext)


    def plot_graph_snapshots(self, folder='images', node_size=100, node_labels=False, cluster=False, ext='.png'):
        """plot the curvature on the graph for each time"""

        if not os.path.isdir(folder):
            os.mkdir(folder)

        print('plot images')
        for i, t in enumerate(tqdm(self.T)):  
            self.plot_graph(i, node_size=node_size, node_labels=node_labels, cluster=cluster)
            plt.title(r'$log_{10}(t)=$'+str(np.around(np.log10(t),2)))
            plt.savefig(folder + '/image_' + str(i) + ext, bbox_inches='tight')
            plt.close()
            

    def plot_graph_3D(G, node_colors=[], edge_colors=[], params=None, save=False, ext='.svg'):
     
        if params==None:
            params = {'elev': 10, 'azim':290}
        elif params!=None and 'elev' not in params.keys():
            params['elev'] = 10
            params['azim'] = 290
        
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if nx.get_node_attributes(G, 'pos') == {}:
            pos = nx.spring_layout(G, dim=3)
        else:
            pos = nx.get_node_attributes(G, 'pos')   
         
        xyz = []
        for i in range(len(pos)):
            xyz.append(pos[i])
            
        xyz = np.array(xyz)
            
        #node colors
        if node_colors=='degree':
            edge_max = max([G.degree(i) for i in range(n)])
            node_colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)] 
        elif nx.get_node_attributes(G, 'color')!={} and node_colors==[]:
            node_colors = nx.get_node_attributes(G, 'color')
            colors = []
            for i in range(n):
                colors.append(node_colors[i])
            node_colors = np.array(colors)    
        else:
            node_colors = 'k'
         
        #edge colors
        if edge_colors!=[]:
            edge_color = plt.cm.cool(edge_colors) 
            width = np.exp(-(edge_colors - np.min(np.min(edge_colors),0))) + 1
            norm = mpl.colors.Normalize(vmin=np.min(edge_colors), vmax=np.max(edge_colors))
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
            cmap.set_array([])    
        else:
            edge_color = ['b' for x in range(m)]
            width = [1 for x in range(m)]
            
        # 3D network plot
        with plt.style.context(('ggplot')):
            
            fig = plt.figure(figsize=(10,7))
            ax = Axes3D(fig)
                       
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=node_colors, s=200, edgecolors='k', alpha=0.7)
               
            for i,j in enumerate(G.edges()): 
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))
                       
                ax.plot(x, y, z, c=edge_color[i], alpha=0.5, linewidth = width[i])
        
        if edge_colors!=[]:    
            fig.colorbar(cmap)        
        ax.view_init(elev = params['elev'], azim=params['azim'])

        ax.set_axis_off()
     
        if save is not False:
            if 'counter' in params.keys():
                fname = G.name + str(params['counter']) + ext
            else:
                fname = G.name + ext
            plt.savefig(fname)
            plt.close('all')       


    def plot_embedding(self, folder='images', ext='.png'):

        if not os.path.isdir(folder):
            os.mkdir(folder)
        
        if not os.path.isdir(folder):
            os.mkdir(folder)
        
        node_colors = list(nx.get_node_attributes(self.G, 'color').values())
       
        A_weight = self.run_embedding(weight='weight')

        for i in range(len(self.Y)):
            plt.figure(figsize=(10,7))
            if len(node_colors)>0:
                plt.scatter(self.Y[i][:, 0], self.Y[i][:, 1], c=node_colors)  
                plt.scatter(A_weight[:, 0], A_weight[:, 1], c=node_colors, alpha=0.5, marker ='+')  
            else:
                plt.scatter(self.Y[i][:, 0], self.Y[i][:, 1])  
                plt.scatter(A_weight[:, 0], A_weight[:, 1], alpha=0.5, marker ='+')  
            plt.axis('tight')
            plt.savefig(os.path.join(folder, 'images_'+str(i) + ext))

    # =============================================================================
    # Functions for saving and loading
    # =============================================================================

    def save_curvature(self, filename = None):
        if not filename:
            filename = self.G.graph.get('name')
        pickle.dump([self.Kappa, self.T], open(filename + '_curvature.pkl','wb'))  

    def load_curvature(self, filename = None):
        if not filename:
            filename = self.G.graph.get('name')
        self.Kappa, self.T = pickle.load(open(filename + '_curvature.pkl','rb'))
        self.n_t = len(self.T)

    def save_clustering(self, filename = None):
        if not filename:
            filename = self.G.graph.get('name')
        pickle.dump([self.G, self.clustering_results, self.labels_gt], open(filename + '_cluster_' + self.cluster_tpe + '.pkl','wb'))
