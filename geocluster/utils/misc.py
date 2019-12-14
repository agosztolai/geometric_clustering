#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:13:18 2019

@author: adamgosztolai
"""
import matplotlib as mpl
#mpl.use('Agg')
import pickle
import pylab as plt
import numpy as np
import networkx as nx
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

# =============================================================================
# Functions for plotting
# =============================================================================

def plot_clustering(gc):
    """plot the clustering results"""
    
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(5,5))
    
    T = np.log10(gc.clustering_results['Markov time'])
    
    if gc.cluster_tpe == 'threshold':
        ax1 = plt.gca()
        ax1.semilogx(T, gc.clustering_results['number_of_communities'], 'C0')
        ax1.set_xlabel('Markov time')
        ax1.set_ylabel('# communities', color='C0')

        ax2 = ax1.twinx()
        ax2.semilogx(T, gc.clustering_results['MI'], 'C1')
        ax2.set_ylabel('Average mutual information', color='C1')
    else:
        #get the times paramters
        n_t = len(gc.clustering_results['ttprime'])

        gs = gridspec.GridSpec(2, 1, height_ratios = [ 1., 0.5])
        gs.update(hspace=0)        

        #make the ttprime matrix
        ttprime = np.zeros([n_t,n_t])
        for i, tt in enumerate(gc.clustering_results['ttprime']):
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
        ax1.plot(T, gc.clustering_results['number_of_communities'],c='C0',label='size',lw=2.)           
        ax1.yaxis.tick_right()
        ax1.tick_params('y', colors='C0')
        ax1.yaxis.set_label_position('right')
        ax1.set_ylabel('Number of clusters', color='C0')
    
        #plot the stability
        ax2 = plt.subplot(gs[1, 0])
        ax2.plot(T, gc.clustering_results['stability'], label=r'$Q$',c='C2')
        #ax2.set_yscale('log') 
        ax2.tick_params('y', colors='C2')
        ax2.set_ylabel('Modularity', color='C2')
        ax2.yaxis.set_label_position('left')
        ax2.set_xlabel(r'$log_{10}(t)$')
        
        #plot the MMI
        ax3 = ax2.twinx()
        ax3.plot(T, gc.clustering_results['MI'],'-',lw=2.,c='C3',label='MI')
        ax3.yaxis.tick_right()
        ax3.tick_params('y', colors='C3')
        ax3.set_ylabel(r'Mutual information', color='C3')
        ax3.axhline(1,ls='--',lw=1.,c='C3')
        ax3.axis([T[0], T[-1], 0,1.1])
        
    plt.savefig('clustering.svg', bbox_inches = 'tight')
    
    
def plot_graph(gc, t, node_size=100, edge_width=2, node_labels=False, cluster=False):
    """plot the curvature on the graph for a given time t"""
    
        
    if 'pos' in gc.G.nodes[0]:
        pos = list(nx.get_node_attributes(gc.G,'pos').values())
    else:
        pos = nx.spring_layout(gc.G)  
    
    if cluster:
        _labels = gc.clustering_results['community_id'][t]
    else:
        _labels = [0] * gc.n
        
    edge_vmin = -1. #-np.max(abs(gc.Kappa[:,t]))
    edge_vmax = 1. #np.max(abs(gc.Kappa[:,t]))    

    plt.figure(figsize = (5,4))
    if len(pos[0])>2:
        pos = np.asarray(pos)[:,[0,2]]
    nodes = nx.draw_networkx_nodes(gc.G, pos=pos, node_size=node_size)#, node_color=_labels, cmap=plt.get_cmap("tab20"))
    edges = nx.draw_networkx_edges(gc.G, pos=pos, width=edge_width, edge_color=gc.Kappa[:, t], edge_vmin=edge_vmin, edge_vmax=edge_vmax, edge_cmap=plt.cm.coolwarm, arrows=False)

    plt.colorbar(edges, label='Edge curvature')

    if node_labels:
        labels_gt={}
        for i in gc.G:
            labels_gt[i] = str(i) + ' ' + str(gc.G.nodes[i]['old_label'])
        nx.draw_networkx_labels(gc.G, pos=pos, labels=labels_gt)

    plt.axis('off')


def plot_edge_curvature(gc):

    plt.figure()
    plt.plot(np.log10(gc.T), gc.Kappa.T, c='C0', lw=0.5)
    plt.axvline(np.log10(gc.T[np.argmax(np.std(gc.Kappa.T,1))]), c='r', ls='--')
    plt.axhline(1, ls='--', c='k')
    plt.axhline(0, ls='--', c='k')
    plt.xlabel('log(time)')
    plt.ylabel('edge OR curvature')
    plt.savefig('edge_curvatures.png')


def plot_graph_snapshots(gc, folder='images', node_size=100, node_labels=False, cluster=False):
    """plot the curvature on the graph for each time"""

    plot_edge_curvature(gc)

    #create folder if not already there
    if not os.path.isdir(folder):
        os.mkdir(folder)

    print('plot images')
    for i, t in enumerate(tqdm(gc.T)):  
        plot_graph(gc, i, node_size=node_size, node_labels=node_labels, cluster=cluster)
        plt.title(r'$log_{10}(t)=$'+str(np.around(np.log10(t),2)))
        plt.savefig(folder + '/clustering_' + str(i) + '.png', bbox_inches='tight')
        plt.close()
        

def plot_graph_3D(G, node_colors=[], edge_colors=[], params=None, save=False):
 
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
            fname = G.name + str(params['counter']) + '.svg'
        else:
            fname = G.name + '.svg'
        plt.savefig(fname)
        plt.close('all')       


def plot_embedding(gc):
    
    node_colors = nx.get_node_attributes(gc.G, 'color')
    colors = []
    for i in range(gc.n):
        colors.append(node_colors[i])
    node_colors = np.array(colors)
    
    for i in range(len(gc.Y)):
        plt.figure(figsize=(10,7))
        plt.scatter(gc.Y[i][:, 0], gc.Y[i][:, 1], c=colors)  
        plt.axis('tight')
        plt.savefig(gc.G.name +  str(i) + 'embedding.svg')

# =============================================================================
# Functions for saving and loading
# =============================================================================

def save_curvature(gc, filename = None):
    if not filename:
        filename = gc.G.graph.get('name')
    pickle.dump([gc.Kappa, gc.T], open(filename + '_curvature.pkl','wb'))  
#    nx.write_gpickle(gc.G, gc.G.graph.get('name') + ".gpickle")

def load_curvature(gc, filename = None):
    if not filename:
        filename = gc.G.graph.get('name')
    gc.Kappa, gc.T = pickle.load(open(filename + '_curvature.pkl','rb'))

def save_clustering(gc, filename = None):
    if not filename:
        filename = gc.G.graph.get('name')
    pickle.dump([gc.G, gc.clustering_results, gc.labels_gt], open(filename + '_cluster_' + gc.cluster_tpe + '.pkl','wb'))