import networkx as nx
import numpy as np

def generate_graph(tpe='SM', params= {}):

    pos = [] 

    if tpe == 'SM':
        G = nx.newman_watts_strogatz_graph(params['n'], params['k'], params['p'])
        
    elif tpe == 'ER':
        G = nx.erdos_renyi_graph(params['n'], params['p'], seed=params['seed'])

    elif tpe == 'barbell':
        G = nx.barbell_graph(params['m1'], params['m2'])

    elif tpe == 'tree':
        G = nx.balanced_tree(params['r'], params['h'])
         
    elif tpe == 'grid' or tpe == 'grid_rect':
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=False)
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')

        pos = {}
        for i in G:
            pos[i] = np.array(G.node[i]['old_label'])
         

    elif tpe == 'torus':
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=True)
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')

        pos = {}
        for i in G:
            pos[i] = np.array(G.node[i]['old_label'])
         
    elif tpe == '2grid':

        F = nx.grid_2d_graph(params['n'], params['n'], periodic = False)
        F = nx.convert_node_labels_to_integers(F, label_attribute='old_label')

        pos = {}
        for i in F:
            pos[i] = np.array(F.node[i]['old_label'])
            
        H = nx.grid_2d_graph(params['n'], params['n'], periodic = False)
        H = nx.convert_node_labels_to_integers(H, first_label=len(F), label_attribute='old_label')

        for i in H:
            pos[i] = np.array(H.node[i]['old_label']) + np.array([params['n']+5, 0])

        G = nx.compose(F, H)

        """
        G.add_node(len(G))
        pos[len(G)-1] = np.array([params['n']+2.5, (params['n']-1)/2.])

        G.add_edge(int(params['n']**2-params['n']/2+1), len(G)-1)
        G.add_edge(int(params['n']**2-params['n']/2),len(G)-1)
        G.add_edge(int(params['n']**2-params['n']/2-1),len(G)-1)
        G.add_edge(len(G)-1, int(params['n']**2 +params['n']/2+1))
        G.add_edge(len(G)-1, int(params['n']**2 +params['n']/2))
        G.add_edge(len(G)-1, int(params['n']**2 +params['n']/2-1))
        """

        G.add_edge(int(params['n']**2-params['n']/2+1), int(params['n']**2 +params['n']/2+1))
        G.add_edge(int(params['n']**2-params['n']/2), int(params['n']**2 +params['n']/2))
        G.add_edge(int(params['n']**2-params['n']/2-1), int(params['n']**2 +params['n']/2-1))

    elif tpe == 'karate':
        G = nx.karate_club_graph()

        for i,j in G.edges:
            G[i][j]['weight']= 1.

        for i in G:
            G.node[i]['old_label'] =  str(i) + ' ' + G.node[i]['club']
        
    elif tpe == 'miserable':
        G = nx.read_gml('../datasets/lesmis.gml')
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')

        for i,j in G.edges:
            G[i][j]['weight']= 1.
    
    elif tpe == 'dolphin':
        G = nx.read_gml('../datasets/dolphins.gml')
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label' )
        
        for i,j in G.edges:
            G[i][j]['weight']= 1.

    elif tpe == 'football':
        G = nx.read_gml('../datasets/football.gml')
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label' )

    elif tpe == 'SBM' or tpe == 'SBM_2':
        #import SBM as sbm
        G = nx.stochastic_block_model(params['sizes'],np.array(params['probs'])/params['sizes'][0], seed=params['seed'])
        for i in G:
            G.node[i]['old_label'] = str(G.node[i]['block'])
        for i,j in G.edges:
            G[i][j]['weight'] = 1.
        
        G = nx.convert_node_labels_to_integers(G, label_attribute='labels_orig')
        #G,community_labels= sbm.SBM_graph(params['n'], params['n_comm'], params['p'])
        
    elif tpe == 'powerlaw':
        G = nx.powerlaw_cluster_graph(params['n'], params['m'], params['p'])

    elif tpe == 'geometric':
        G = nx.random_geometric_graph(params['n'], params['p'])
        
    elif tpe == 'netscience':

        G_full = nx.read_gml('../datasets/netscience.gml')

        #use integer for labels
        G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='old_label')

        #get largest connected component
        largest_cc = sorted(max(nx.connected_components(G_full), key=len))
        G = G_full.subgraph(largest_cc)
        
        #relabel the integers 
        G = nx.convert_node_labels_to_integers(G)

    elif tpe == 'powergrid':
        edges    = np.genfromtxt('../datasets/UCTE_edges.txt')
        location = np.genfromtxt('../datasets/UCTE_nodes.txt')
        posx = location[:,1]
        posy = location[:,2]
        pos  = {}

        edges = np.array(edges,dtype=np.int32)
        G = nx.Graph() #empty graph
        G.add_edges_from(edges) #add edges

        G = nx.convert_node_labels_to_integers(G, label_attribute = 'old_label' )
        

        #create the position vector for plotting
        for i in G.nodes():
            pos[i] = [posx[G.node[i]['old_label']-1],posy[G.node[i]['old_label']-1]]
            #pos[i]= [posx[i-1],posy[i-1]]
            
    elif tpe == 'email':

        edges = np.loadtxt('../datasets/email-Eu-core.txt')
        G = nx.DiGraph()
        G.add_edges_from(edges) #add edges

        G = nx.convert_node_labels_to_integers(G)

    elif tpe == 'tutte':
        G = nx.tutte_graph()
    
    elif tpe == 'krackhardt':
        G = nx.Graph(nx.krackhardt_kite_graph())
        for i,j in G.edges:
            G[i][j]['weight']= 1.

    elif tpe == 'frucht':
        G = nx.frucht_graph()
        
    elif tpe == 'scale free':
        G = nx.DiGraph(nx.scale_free_graph(params['n']))

    elif tpe == 'gnr':
        #directed growing network
        G = nx.gnr_graph(params['n'], params['p'])
        
    elif tpe == 'celegans':
        #directed growing network
        from datasets.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../datasets/celegans/')

        for i in G:
            G.node[i]['old_label'] = G.node[i]['labels']

    elif tpe == 'celegans_undirected':
        #directed growing network
        from datasets.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../datasets/celegans/')

        for i in G:
            G.node[i]['old_label'] = G.node[i]['labels']

        G = G.to_undirected()

    elif tpe == 'delaunay-grid':

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
        
    elif tpe == 'Fan':
        G = nx.planted_partition_graph(4, 32, 1/8, 1/8, seed=params['seed'])
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        for edge in G.edges:
            if G.node[edge[0]]['block'] == G.node[edge[1]]['block']:
                G.edges[edge]['weight'] = params['w_in']
            else:
                G.edges[edge]['weight'] = 2 - params['w_in']

    elif tpe == 'delaunay':

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

    return G, pos