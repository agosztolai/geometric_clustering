import networkx as nx
import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
import sklearn.datasets as skd
import sklearn.neighbors as skn
from misc import plot_graph_3D

'''
# =============================================================================
# Library of standard graphs
# =============================================================================

barbell : barbell graph
barbell_asy : asymmetric barbell graph
celegans:  neural network of neurons and synapses in C. elegans
grid : rectangular grid
2grid : 2 rectangular grids connected by a bottleneck
delaunay-grid : Delaunay triangulation of uniformly spaced points
delauney-nonunif : Delaunay triangulation of nonuniform points (2 Gaussians)
dolphin : directed social network of bottlenose dolphins
email : network of email data from a large European research institution
ER : Erdos-Renyi graph
Fan : Fan's benchmark graph
football : 
frucht : 
GN : Girvan-Newman benchmark
gnr : directed growing network  
karate : Zachary's karate club
LFR : Lancichinetti-Fortunato-Radicchi benchmark  
krackhardt : 
miserable : 
netscience
scalefree    
S : S-curve
SM : small-world network
SB
SBM : stochastic block model
swiss-roll : Swiss-roll dataset
torus
tree
tutte
    
'''

def main():
#    generate_graph('S',{'n': 300, 'seed': 0, 'similarity': 'knn', 'k': 10},save=True)
    generate_graph('swiss-roll',{'n': 300, 'noise':1.0, 'similarity': 'knn', 'k': 10, 'elev': 10, 'azim': 280},save=True)

def generate_graph(tpe='SM', params= {}, save=False):

    pos = None
    color = []
    
    if tpe == 'barbell':
        G = nx.barbell_graph(params['m1'], params['m2'])
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
     
    if tpe == 'barbell_noisy':
        G = nx.barbell_graph(params['m1'], params['m2'])
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
        for i,j in G.edges():
            G[i][j]['weight'] = abs(np.random.normal(1,params['noise']))
            
        
    elif tpe == 'barbell_asy':
        A = np.block([[np.ones([params['m1'], params['m1']]), np.zeros([params['m1'],params['m2']])],\
                       [np.zeros([params['m2'],params['m1']]), np.ones([params['m2'],params['m2']])]])
        A = A - np.eye(params['m1'] + params['m2'])
        A[params['m1']-1,params['m1']] = 1
        A[params['m1'],params['m1']-1] = 1
        G = nx.from_numpy_matrix(A)   
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
            
    elif tpe == 'celegans':
        from skd.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../../datasets/celegans/')

        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']

    elif tpe == 'celegans_undirected':
        from skd.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../datasets/celegans/')

        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']

        G = G.to_undirected()        
            
    elif tpe == 'grid':
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=False)
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')

        pos = {}
        for i in G:
            pos[i] = np.array(G.nodes[i]['old_label'])                  
       
    elif tpe == '2grid':

        F = nx.grid_2d_graph(params['n'], params['n'], periodic = False)
        F = nx.convert_node_labels_to_integers(F, label_attribute='old_label')

        pos = {}
        for i in F:
            pos[i] = np.array(F.nodes[i]['old_label'])
            
        H = nx.grid_2d_graph(params['n'], params['n'], periodic = False)
        H = nx.convert_node_labels_to_integers(H, first_label=len(F), label_attribute='old_label')

        for i in H:
            pos[i] = np.array(H.nodes[i]['old_label']) + np.array([params['n']+5, 0])

        G = nx.compose(F, H)
        G.add_edge(int(params['n']**2-params['n']/2+1), int(params['n']**2 +params['n']/2+1))
        G.add_edge(int(params['n']**2-params['n']/2), int(params['n']**2 +params['n']/2))
        G.add_edge(int(params['n']**2-params['n']/2-1), int(params['n']**2 +params['n']/2-1))        

    if tpe == 'delaunay-grid':
        from scipy.spatial import Delaunay
        np.random.seed(0)
        x = np.linspace(0,1,params['n'])

        points = []
        for i in range(params['n']):
            for j in range(params['n']):
                points.append([x[j],x[i]])

        points = np.array(points)

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0],t[1]])
            edge_list.append([t[0],t[2]])
            edge_list.append([t[1],t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()        
            
    elif tpe == 'delaunay-nonunif':
        from scipy.spatial import Delaunay
        np.random.seed(0)
        x = np.linspace(0,1,params['n'])

        points = []
        for i in range(params['n']):
            for j in range(params['n']):
                points.append([x[j],x[i]])

        points = np.array(points)

        gauss_pos = [.5, 0.5]
        gauss_pos2 = [0.7, 0.7]
        gauss_var = [.05,.05]
        new_points = np.random.normal(gauss_pos, gauss_var , [20,2])
        #new_points = np.concatenate( (new_points, np.random.normal( gauss_pos2, gauss_var, [50,2])) )

        for p in new_points:
            if p[0]>0 and p[0]<1. and p[1]>0 and p[1]<1:
                points = np.concatenate( (points, [p,]) )

        #points = np.concatenate( (points, np.random.normal(.5,.1, [200,2])) )

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0],t[1]])
            edge_list.append([t[0],t[2]])
            edge_list.append([t[1],t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()
            
    elif tpe == 'dolphin':
        G = nx.read_gml('../../datasets/dolphins.gml')
        G = nx.convert_node_labels_to_integers(G)     
        for i,j in G.edges:
            G[i][j]['weight']= 1.
    
    elif tpe == 'email':
        edges = np.loadtxt('../../datasets/email-Eu-core.txt').astype(int)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        labels = np.loadtxt('../../datasets/email-Eu-core-department-labels.txt').astype(int)
        for i in G:
            G.nodes[i]['block'] = labels[i]
        
        G = nx.convert_node_labels_to_integers(G)
        
    elif tpe == 'ER':
        G = nx.erdos_renyi_graph(params['n'], params['p'], seed=params['seed'])  
    
    elif tpe == 'Fan':
        G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], params['seed'])   
        
        for i,j in G.edges:
            if G.nodes[i]['block'] == G.nodes[j]['block']:
                G[i][j]['weight'] = params['w_in']
            else:
                G[i][j]['weight'] = 2 - params['w_in']
                
        labels_gt = []
        for i in range(params['l']):
            labels_gt = np.append(labels_gt,i*np.ones(params['g']))
            
        for n in G.nodes:
            G.nodes[n]['block'] = labels_gt[n-1]   
            
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label') 
        
    elif tpe == 'football':
        G = nx.read_gml('../datasets/football.gml')
        
    elif tpe == 'frucht':
        G = nx.frucht_graph()    
      
    elif tpe == 'GN':
        G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], seed=params['seed'])
        
        labels_gt = []
        for i in range(params['l']):
            labels_gt = np.append(labels_gt,i*np.ones(params['g']))
            
        for n in G.nodes:
                G.nodes[n]['block'] = labels_gt[n-1]    

    elif tpe == 'gnr':
        #directed growing network
        G = nx.gnr_graph(params['n'], params['p'])

    elif tpe == 'karate':
        G = nx.karate_club_graph()

        for i,j in G.edges:
            G[i][j]['weight']= 1.

        for i in G:
            G.nodes[i]['block'] =  str(i) + ' ' + G.nodes[i]['club']
     
    elif tpe == 'LFR':
        import os
        command = params['scriptfolder'] + \
        " -N " + str(params['n']) + \
        " -t1 " + str(params['tau1']) + \
        " -t2 " + str(params['tau2']) + \
        " -mut " + str(params['mu']) + \
        " -muw " + str(0.5) + \
        " -maxk " + str(params['n']) + \
        " -k " + str(params['k']) + \
        " -name data"
            
        os.system(command)
        G = nx.read_weighted_edgelist('data.nse', nodetype=int, encoding='utf-8')
        for e in G.edges:
            G.edges[e]['weight'] = 1
                
        labels = np.loadtxt('data.nmc',usecols=1,dtype=int)    
        for n in G.nodes:
            G.nodes[n]['block'] = labels[n-1]    
        
    elif tpe == 'krackhardt':
        G = nx.Graph(nx.krackhardt_kite_graph())
        for i,j in G.edges:
            G[i][j]['weight']= 1.    
        
    elif tpe == 'miserable':
        G = nx.read_gml('../datasets/lesmis.gml')

        for i,j in G.edges:
            G[i][j]['weight']= 1.
            
    elif tpe == 'netscience':
        G_full = nx.read_gml('../../datasets/netscience.gml')
        G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='old_label')
        largest_cc = sorted(max(nx.connected_components(G_full), key=len))
        G = G_full.subgraph(largest_cc)
        G = nx.convert_node_labels_to_integers(G)   

    elif tpe == 'powerlaw':
        G = nx.powerlaw_cluster_graph(params['n'], params['m'], params['p'])

    elif tpe == 'geometric':
        G = nx.random_geometric_graph(params['n'], params['p'])
            
    elif tpe == 'powergrid':
        edges    = np.genfromtxt('../datasets/UCTE_edges.txt')
        location = np.genfromtxt('../datasets/UCTE_nodes.txt')
        posx = location[:,1]
        posy = location[:,2]
        pos  = {}

        edges = np.array(edges,dtype=np.int32)
        G = nx.Graph() #empty graph
        G.add_edges_from(edges) #add edges        

        #create the position vector for plotting
        for i in G.nodes():
            pos[i] = [posx[G.nodes[i]['old_label']-1],posy[G.nodes[i]['old_label']-1]]
            #pos[i]= [posx[i-1],posy[i-1]]

    elif tpe == 'S':   
        pos, color = skd.samples_generator.make_s_curve(params['n'], random_state=params['seed'])

    elif tpe == 'scale-free':
        G = nx.DiGraph(nx.scale_free_graph(params['n']))                      
         
    elif tpe == 'SBM' or tpe == 'SBM_2':
        G = nx.stochastic_block_model(params['sizes'],np.array(params['probs'])/params['sizes'][0], seed=params['seed'])
        for i,j in G.edges:
            G[i][j]['weight'] = 1.
        
        G = nx.convert_node_labels_to_integers(G, label_attribute='labels_orig')        

    elif tpe == 'SM':
        G = nx.newman_watts_strogatz_graph(params['n'], params['k'], params['p'])    
        
    elif tpe == 'swiss-roll':
        pos, color = skd.make_swiss_roll(n_samples=params['n'], noise=params['noise'], random_state=None)    

    elif tpe == 'torus':
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=True)

        pos = {}
        for i in G:
            pos[i] = np.array(G.nodes[i]['old_label'])

    elif tpe == 'tree':
        G = nx.balanced_tree(params['r'], params['h'])      
        
    elif tpe == 'tutte':
        G = nx.tutte_graph()  
                          
        
# =============================================================================
#  Set graph attributes
# =============================================================================   
    
    if 'similarity' in params.keys():
        similarity = params['similarity']
        if similarity=='euclidean' or similarity=='minkowski':
            A = squareform(pdist(pos, similarity))
        elif similarity=='knn':
            A = skn.kneighbors_graph(pos, params['k'], mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
            A = A.todense()
        elif similarity=='radius':
            A = skn.radius_neighbors_graph(pos, params['radius'], mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
            A = A.todense()
        elif similarity=='heat':    
            A = squareform(pdist(pos, 'euclidean'))
            A = sp.exp(-A ** 2 / params['s'] ** 2)
            np.fill_diagonal(A, 0)
            
        G = nx.from_numpy_matrix(A)  
        for i in G:
            G.nodes[i]['block'] = pos[i]
            if color!=[]:
                G.nodes[i]['color'] = color[i]
    
    assert nx.is_connected(G), 'Graph is disconnected!'
    G.graph['name'] = tpe
    
    if pos is None:
        pos = nx.spring_layout(G)
        
    attrs = {}
    for i in G.nodes:
        if color==[]:
            attrs[i] = {'pos': pos[i], 'color': 'k'}  
        else:
            attrs[i] = {'pos': pos[i], 'color': color[i]} 
        
    nx.set_node_attributes(G, attrs)    
        
    if 'block' in G.nodes[0]:   
        for i in G:
            G.nodes[i]['old_label'] = str(G.nodes[i]['block']) 
        
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')    
            
<<<<<<< HEAD
    if save:
        plot_graph_3D(G, params=params, save=True)   

    return G

         
if __name__ == "__main__":
	main()        
=======
    return G
>>>>>>> b676661b99fc7fe8ab58d404c78cf04b2b80a5b8
