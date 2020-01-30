#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import numpy as np
import scipy as sc
from tqdm import tqdm
import networkx as nx
import pickle as pickle
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from multiprocessing import Pool
from functools import partial
from scipy.sparse.csgraph import floyd_warshall
from sklearn.utils import check_symmetric

from .curvature import mx_comp, K_ij, K_all, K_all_gpu
from .clustering import cluster_threshold
from .embedding import signed_laplacian, SpectralEmbedding


class GeoCluster(object): 

    def __init__(self, G, T=np.logspace(0,1,10), laplacian_tpe='normalized',
                 use_spectral_gap=True):

        self.G = G
        self.A = check_symmetric(nx.adjacency_matrix(self.G, weight='weight'))
        self.n = len(G.nodes)
        self.e = len(G.edges)
        self.use_spectral_gap = use_spectral_gap
        self.laplacian_tpe = laplacian_tpe
        
        #time vector
        self.n_t = len(T) - 1
        self.T = T   


    def compute_OR_curvatures(self, with_weights=False, GPU=False, workers=2, 
                              cutoff=1.0, lamb=0, save=True):
        """Edge curvature matrix"""    
        
        print('\nGraph: ' + self.G.graph['name'])
        
        L = construct_laplacian(self.G, self.laplacian_tpe, self.use_spectral_gap) #Laplacian matrix 
        dist = compute_distance_geodesic(self.A) #Geodesic distance matrix
       
        print('\nCompute curvature at each markov time')
        self.Kappa = np.ones([self.e, self.n_t])
        for it in tqdm(range(self.n_t)): 

            mxs = list(np.eye(self.n))  # create delta at each node
            
            with Pool(processes = workers) as p_mx: 
                mxs = p_mx.map_async(partial(mx_comp, L, self.T[it+1] - self.T[it]), mxs).get()
            
            if not GPU:
                with Pool(processes = workers) as p_kappa:  
                    self.Kappa[:, it] = p_kappa.map_async(partial(K_ij, mxs, dist, lamb, cutoff,  with_weights, list(self.G.edges())), range(self.e)).get()
            else: 
                for i in range(len(mxs)):
                    self.Kappa[:,it] = K_all_gpu(mxs, dist, lamb, self.G, with_weights=with_weights)  

            if all(self.Kappa[:,it]>0):
                print('All edges have positive curvatures, so stopping the computations')
                break
    
            if save:
                self.save_curvature(t_max = it)


    def run_clustering(self,cluster_tpe='threshold', cluster_by='curvature'):
        """Clustering of curvature weigthed graphs"""
        
        self.cluster_tpe = cluster_tpe
        self.labels_gt = [int(self.G.nodes[i]['block']) for i in self.G.nodes if 'block' in self.G.nodes[0]]

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
            stability.n_mi = 10  #if all_mi = False, number of top Louvain run to use for MI        
            stability.n_processes_louv = 10 #number of cpus 
            stability.n_processes_mi = 10 #number of cpus

            stabilities, nComms, MIs, labels = [], [], [], []
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

            ttprime = stability.compute_ttprime(labels, nComms, self.T[:np.shape(self.Kappa)[1]])

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
        
        T = np.log10(self.clustering_results['Markov time'])[:len(self.clustering_results['ttprime'])]
        
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
            #ax0.axis([T[0],T[-1],T[0],T[-1]])
            ax0.axis([-1,T[-1],T[0],T[-1]])

            #plot the number of clusters
            ax1 = ax0.twinx()
            ax1.plot(T, self.clustering_results['number_of_communities'],c='C0',label='size',lw=2.)           
            ax1.yaxis.tick_right()
            ax1.tick_params('y', colors='C0')
            ax1.yaxis.set_label_position('right')
            ax1.set_ylabel('Number of clusters', color='C0')
            ax1.set_ylim(0,500) 
            
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
            #ax3.axis([T[0], T[-1], 0,1.1])
            ax3.axis([-1, T[-1], 0.7,1.1])
            
        plt.savefig('clustering'+ext, bbox_inches = 'tight')
        
        
    def plot_graph(self, t, node_size=20, edge_width=1, node_labels=False, 
                   cluster=False, node_colors=None, figsize=(10, 7)):
        """plot the curvature on the graph for a given time t"""
        
            
        if 'pos' in self.G.nodes[1]:
            pos = list(nx.get_node_attributes(self.G,'pos').values())
        
        else:
            pos = nx.spring_layout(self.G)  
        
        if len(pos[0])>2:
            pos = np.asarray(pos)[:,[0,2]]
            
        plt.figure(figsize = figsize)
        
        if cluster:
            node_colors = self.clustering_results['community_id'][t]
            cmap = plt.get_cmap("tab20")
        
        elif node_colors is not None:
            cmap = plt.cm.coolwarm
        
        else:
            node_colors = [0] * self.n
            cmap = plt.get_cmap("tab20")
          
        edge_vmin = -1. #-np.max(abs(self.Kappa[:,t]))
        edge_vmax = 1. #np.max(abs(self.Kappa[:,t])) 
            
        nx.draw_networkx_nodes(self.G, pos=pos, node_size=node_size, node_color=node_colors, cmap=cmap) 
        nx.draw_networkx_edges(self.G, pos=pos, width=edge_width, edge_color=self.Kappa[:, t], 
                               edge_vmin=edge_vmin, edge_vmax=edge_vmax, edge_cmap=plt.cm.coolwarm, 
                               alpha=0.5, arrows=False)

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


    def plot_edge_curvature(self, ext='.svg', density=False, zeros=True, 
                            log=True, shift_origin=0.4, save=True, filename=''):
        
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[3, 1], height_ratios=[3, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])

        gs.update(wspace=0.00)
        gs.update(hspace=0)

        ax1.plot(np.log10(self.T[:-1]), self.Kappa.T, c='C0', lw=0.5)
        ax1.axhline(1, ls='--', c='k')
        ax1.axhline(0, ls='--', c='k')
        
        if log:
            ax1.set_yscale('symlog')
            
        ax1.set_ylabel('log(edge OR curvature)')
        ax1.set_ylim([np.min(self.Kappa),1])
        ax1.set_xlim([np.log10(self.T[0]), np.log10(self.T[-1])])
        #ax1.get_xaxis().set_visible(False)
        
        if density:
            
            from sklearn.neighbors import KernelDensity
            
            #find minima
            mins = [ np.min(self.Kappa[i]) for i in range(self.Kappa.shape[0]) ]
            mins = np.array(mins)
            inds =  np.array([ np.argmin(self.Kappa[i]) for i in range(self.Kappa.shape[0]) ])
            inds = inds[mins<0]
            mins = mins[mins<0][:, np.newaxis]
    
            if len(inds)>0:
            
                ax2 = fig.add_subplot(gs[1, 0])
            
                bw = self.Kappa.shape[0]**(-1./(1+4)) #Scott's rule
                kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.log10(self.T[inds])[:, np.newaxis])
                Tind = np.linspace(np.log10(self.T[0]), np.log10(self.T[-2]), 100)[:, np.newaxis]
                log_dens = kde.score_samples(Tind)
                ax2.plot(Tind[:, 0], np.exp(log_dens), color='navy', linestyle='-')
                
                ax2.scatter(np.log10(self.T[inds]), np.zeros_like(inds))
                ax2.tick_params(axis='x', which='both', left=False, top=False, labelleft=False)
                ax2.set_ylim([-0.1,1])
                ax2.set_xlabel('log(time)')
        
                kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(mins)
                minind = np.linspace(np.min(self.Kappa),1,100)[:, np.newaxis]

                log_dens = kde.score_samples(minind) 

        elif zeros:
            shift = int(shift_origin*len(self.T))
            Kappa = self.Kappa[:, shift:-2]

            t_mins =  self.T[shift + np.array([ np.argmin(abs(Kappa[i])) for i in range(Kappa.shape[0]) ])]

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(np.log10(t_mins), bins = len(self.T)-shift-1, range = (np.log10(self.T[0]), np.log10(self.T[-1])), log=True)
            ax2.set_xlim([np.log10(self.T[0]), np.log10(self.T[-1])])

        if save:
            plt.savefig(filename + 'edge_curvatures' + ext)
            
        return fig


    def plot_graph_snapshots(self, folder='images', filename = 'image', 
                             node_size=30, edge_width=10, node_labels=False, 
                             cluster=False, ext='.svg', figsize=(5,4)):
        """plot the curvature on the graph for each time"""

        if not os.path.isdir(folder):
            os.mkdir(folder)

        print('plot images')

        for i in tqdm(range(self.n_t)):
            self.plot_graph(i, node_size=node_size, node_labels=node_labels, cluster=cluster, edge_width=edge_width, figsize=figsize)
            plt.title(r'$log_{10}(t)=$' + str(np.around(np.log10(self.T[i]),2)))
            plt.savefig(folder + '/' +  filename + '%03d' % i + ext, bbox_inches='tight')
            plt.close()
            

    def plot_graph_3D(G, node_colors=[], edge_colors=[], params=None, save=False, ext='.svg'):
     
        if params==None:
            params = {'elev': 10, 'azim':290}
        elif params!=None and 'elev' not in params.keys():
            params['elev'] = 10
            params['azim'] = 290
        
        n, m = G.number_of_nodes(), G.number_of_edges() 
        
        if nx.get_node_attributes(G, 'pos') == {}:
            pos = nx.spring_layout(G, dim=3)
        else:
            pos = nx.get_node_attributes(G, 'pos')   
         
        xyz = np.array([pos[i] for i in range(len(pos))])
            
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


    def plot_embedding(self, folder='embedding_images', ext='.svg'):

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
            
            plt.savefig(os.path.join(folder, 'images_' + str(i) + ext))

    # =============================================================================
    # Functions for saving and loading
    # =============================================================================

    def save_curvature(self, t_max = None, filename = None):
        if filename is None:
            filename = self.G.graph.get('name')
        if t_max is None:
            pickle.dump([self.Kappa, self.T], open(filename + '_curvature.pkl','wb'))  
        else:
            pickle.dump([self.Kappa[:, :t_max], self.T[:t_max]], open(filename + '_curvature.pkl','wb'))  


    def load_curvature(self, filename = None):
        if not filename:
            filename = self.G.graph.get('name')
        self.Kappa, self.T = pickle.load(open(filename + '_curvature.pkl','rb'))
        self.n_t = len(self.T)


    def save_clustering(self, filename = None):
        if not filename:
            filename = self.G.graph.get('name')
        pickle.dump([self.G, self.clustering_results, self.labels_gt], open(filename + '_cluster_' + self.cluster_tpe + '.pkl','wb'))


    def load_clustering(self, filename = None):
        if not filename:
            filename = self.G.graph.get('name')
            
        self.G, self.clustering_results, self.labels_gt = pickle.load(open(filename + '_cluster_' + self.cluster_tpe + '.pkl','rb'))


    def save_embedding(self, filename = None):
        pickle.dump([self.G, self.Y], open(filename + '_embed.pkl','wb'))


def compute_node_curvature(G, Kappa):
    """Node curvatures from the adjacent edge curvatures"""

    B = nx.incidence_matrix(G).toarray() #incidence matrix with only ones (no negative values)
    Dinv = np.diag(1./B.sum(1)) # inverse degree matrix

    Kappa_node = Dinv.dot(B).dot(Kappa)
    
    return Kappa_node


def compute_distance_geodesic(A):
    """Geodesic distance matrix"""
        
    dist = floyd_warshall(A, directed=True, unweighted=False)
        
    return dist
  
    
def construct_laplacian(G, laplacian_tpe='normalized', use_spectral_gap=False): 
    """Laplacian matrix"""
                
    if laplacian_tpe == 'normalized':
        # degrees = np.array(self.A.sum(1)).flatten()
        degrees = np.array([G.degree[i] for i in G.nodes])
        L = sc.sparse.csr_matrix(nx.laplacian_matrix(G).toarray().dot(np.diag(1./degrees)))

    elif laplacian_tpe == 'combinatorial':
        L = sc.sparse.csr_matrix(1.*nx.laplacian_matrix(G)) #combinatorial Laplacian

    elif laplacian_tpe == 'signed_normalized':
        L = signed_laplacian(G, normed=True, return_diag=True)

    if use_spectral_gap:
        L /= abs(sc.sparse.linalg.eigs(L, which='SM', k=2)[0][1])
            
    return L     