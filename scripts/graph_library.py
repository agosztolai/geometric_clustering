import numpy as np
import networkx as nx
from sklearn.utils import check_symmetric
import sklearn.datasets as skd
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import os
import sklearn.neighbors as skn
import yaml as yaml

'''
# =============================================================================
# Library of standard graphs
# =============================================================================
'''

class GraphGen(object): 

    def __init__(self, whichgraph='barbell', nsamples=1, paramsfile='graph_params.yaml',
                 outfolder=[], plot=True):

        self.paramsfile = paramsfile
        self.whichgraph = whichgraph
        self.color = []
        self.pos = None
        self.plot = plot

        self.params = yaml.load(open(paramsfile,'rb'), Loader=yaml.FullLoader)[whichgraph]
        self.nsamples = nsamples
        if outfolder==[]:            
            self.outfolder = './' + whichgraph
        if 'similarity' not in self.params:
            self.params['similarity'] = None

    def generate(self, dim=2, symmetric=True):
        
        print('\nGraph: ' + self.whichgraph)
        print('\nParameters:', self.params)
        
        #create a folder
        #if not os.path.isdir(self.outfolder):
        #    os.mkdir(self.outfolder)
        self.outfolder = '.'
                
        self.params['counter'] = 0    
        while self.params['counter'] < self.nsamples:
            if 'seed' in self.params.keys():
                self.params['seed'] += 1
            
            try:
                #generate graph
                self.G, self.tpe = graphs(self.whichgraph, self.params)

                #compute similarity matrix if not assigned    
                if self.tpe == 'pointcloud':
                    if self.params['similarity'] != None:
                        A = self.similarity_matrix(symmetric)
                        self.A = A
                        G1 = nx.from_numpy_matrix(A)
                        for i in self.G:
                            G1.nodes[i]['pos'] = self.G.nodes[i]['pos']
                            G1.nodes[i]['color'] = self.G.nodes[i]['color']                    
                        self.G = G1  
                    else:
                        print('Define similarity measure!')
                        break
                    
                #compute positions if not assigned    
                elif self.tpe =='graph':
                    if 'pos' not in self.G.nodes[1]:
                        pos = nx.spring_layout(self.G, dim=dim, weight='weight')
                        for i in self.G:
                            self.G.nodes[i]['pos'] = pos[i]
                            
                #this is for compatibility with PyGenStability
                if 'block' in self.G.nodes[1]:
                    for i in self.G:
                        self.G.nodes[i]['old_label'] = str(self.G.nodes[i]['block'])
                    self.G = nx.convert_node_labels_to_integers(self.G, label_attribute='old_label') 
                    
                #check if graph is connected    
                if nx.is_connected(self.G):
                    
                    #save
                    #fname = self.whichgraph + '_' + str(self.params['counter'])
                    #nx.write_gpickle(self.G, self.outfolder + '/' + fname + "_.gpickle")
                    
                    #plot 2D graph or 3D graph
                    #if self.plot and len(self.G.nodes[1]['pos'])==3:
                    #    fig = plot_graph_3D(self.G, node_colors='custom', params=self.params)  
                    #    fig.savefig(self.outfolder  + '/' + fname + '.svg')
                    #elif self.plot and len(self.G.nodes[1]['pos'])==2:
                    #    fig = plot_graph(self.G, node_colors='cluster')  
                    #    fig.savefig(self.outfolder + '/' + fname + '.svg')
                        
                    self.params['counter'] += 1    
                else:
                    print('Graph is disconnected')
                    
            except Exception as e:
                print('Graph generation failed because ' + str(e) )
                self.params['counter'] = self.nsamples + 1
                
            self.G.graph['name'] = self.whichgraph 
    # =============================================================================
    # similarity matrix
    # =============================================================================
    def similarity_matrix(self, symmetric=True):
        
        n = self.G.number_of_nodes()

        pos = nx.get_node_attributes(self.G,'pos')
        pos = np.reshape([pos[i] for i in range(n)],(n,len(pos[0])))

        color = nx.get_node_attributes(self.G,'color')
        color = [color[i] for i in range(n)]

        params = self.params 

        sim = params['similarity']
        if sim=='euclidean' or sim=='minkowski':
            A = squareform(pdist(pos, sim))
        
        elif sim=='knn':
            A = skn.kneighbors_graph(pos, params['k'], mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
            A = A.todense()
        
        elif sim=='radius':
            A = skn.radius_neighbors_graph(pos, params['radius'], mode='connectivity', metric='minkowski', p=2, metric_params=None, n_jobs=-1)
            A = A.todense()
        
        elif sim=='rbf':    
            gamma_ = (params['gamma']
                               if 'gamma' in params.keys() else 1.0 / pos.shape[1])
            A = rbf_kernel(pos, gamma=gamma_)

        if symmetric==True:
            A = check_symmetric(A)

        return A
     
# =============================================================================
# graphs
# =============================================================================
def graphs(whichgraph, params):
    
    G = nx.Graph()
    
    if whichgraph == 'barbell':
        tpe = 'graph'
        G = nx.barbell_graph(params['m1'], params['m2'])
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
            
    elif whichgraph == 'barbell_noisy':
        tpe = 'graph'
        G = nx.barbell_graph(params['m1'], params['m2'])
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])
        for i,j in G.edges():
            G[i][j]['weight'] = abs(np.random.normal(1,params['noise']))
                
    elif whichgraph == 'barbell_asy':
        tpe = 'graph'
        A = np.block([[np.ones([params['m1'], params['m1']]), np.zeros([params['m1'],params['m2']])],\
                       [np.zeros([params['m2'],params['m1']]), np.ones([params['m2'],params['m2']])]])
        A = A - np.eye(params['m1'] + params['m2'])
        A[params['m1']-1,params['m1']] = 1
        A[params['m1'],params['m1']-1] = 1
        G = nx.from_numpy_matrix(A)   
        for i in G:
            G.nodes[i]['block'] = np.mod(i,params['m1'])

    elif whichgraph == 'complete':
        tpe ='graph'
        G = nx.complete_graph(params['n'])

    elif whichgraph == 'celegans':
        tpe = 'graph'
        from datasets.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = 'datasets/celegans/')
        
        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']
            
    elif whichgraph == 'celegans_undirected':
        tpe = 'graph'
        from datasets.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = 'datasets/celegans/')
        
        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']
            
        G = G.to_undirected()        
        
    elif whichgraph == 'grid':
        tpe = 'graph'
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=False)
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
        pos = {}
        for i in G:
            pos[i] = np.array(G.nodes[i]['old_label'])                  
            
    elif whichgraph == '2grid': 
        tpe = 'graph'          
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
        
    elif whichgraph == 'delaunay-grid':
        tpe = 'graph'
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
        
    elif whichgraph == 'delaunay-nonunif':
        tpe = 'graph'
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
        
    elif whichgraph == 'dolphin':
        tpe = 'graph'
        G = nx.read_gml('datasets/dolphins.gml')
        G = nx.convert_node_labels_to_integers(G)     
        for i,j in G.edges:
            G[i][j]['weight']= 1.

    elif whichgraph == 'email':
        tpe = 'graph'
        edges = np.loadtxt('../../datasets/email-Eu-core.txt').astype(int)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        labels = np.loadtxt('../../datasets/email-Eu-core-department-labels.txt').astype(int)
        for i in G:
            G.nodes[i]['block'] = labels[i]
            
        G = nx.convert_node_labels_to_integers(G)
            
    elif whichgraph == 'ER':
        tpe = 'graph'
        G = nx.erdos_renyi_graph(params['n'], params['p'], seed=params['seed'])  
    
    elif whichgraph == 'Fan':
        tpe = 'graph'
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
    
    elif whichgraph == 'football':
        tpe = 'graph'
        G = nx.read_gml('datasets/football.gml')
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
    elif whichgraph == 'frucht':
        tpe = 'graph'
        G = nx.frucht_graph()    
      
    elif whichgraph == 'GN':
        tpe = 'graph'
        G = nx.planted_partition_graph(params['l'], params['g'], params['p_in'], params['p_out'], seed=params['seed'])
            
        labels_gt = []
        for i in range(params['l']):
            labels_gt = np.append(labels_gt,i*np.ones(params['g']))
            
        for n in G.nodes:
            G.nodes[n]['block'] = labels_gt[n-1]    
    
    elif whichgraph == 'gnr':
        tpe = 'graph'
        #directed growing network
        G = nx.gnr_graph(params['n'], params['p'])

    elif whichgraph == 'karate':
        tpe = 'graph'
        G = nx.karate_club_graph()
        
        for i,j in G.edges:
            G[i][j]['weight']= 1.
    
        for i in G:
            if G.nodes[i]['club'] == 'Mr. Hi':
                G.nodes[i]['block'] = 0
                G.nodes[i]['color'] = 0
            else:
                G.nodes[i]['block'] = 1
                G.nodes[i]['color'] = 1
    
    elif whichgraph == 'LFR':
        tpe = 'graph'        
        command = params['scriptfolder'] + \
        " -N " + str(params['n']) + \
        " -t1 " + str(params['tau1']) + \
        " -t2 " + str(params['tau2']) + \
        " -mut " + str(params['mu']) + \
        " -muw " + str(params['mu']) + \
        " -maxk " + str(params['n']) + \
        " -k " + str(params['k']) + \
        " -name " + params['outfolder'] + "data"
        
        os.system(command)
        G = nx.read_weighted_edgelist(params['outfolder'] +'data.nse', nodetype=int, encoding='utf-8')
        for e in G.edges:
            G.edges[e]['weight'] = 1
            
        labels = np.loadtxt(params['outfolder'] +'data.nmc',usecols=1,dtype=int)    
        for n in G.nodes:
            G.nodes[n]['block'] = labels[n-1]    
    
    elif whichgraph == 'krackhardt':
        tpe = 'graph'
        G = nx.Graph(nx.krackhardt_kite_graph())
        for i,j in G.edges:
            G[i][j]['weight']= 1.    
    
    elif whichgraph == 'miserable':
        tpe = 'graph'
        G = nx.read_gml('datasets/lesmis.gml')
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
        for i,j in G.edges:
            G[i][j]['weight']= 1.
            
    elif whichgraph == 'netscience':
        tpe = 'graph'
        G_full = nx.read_gml('datasets/netscience.gml')
        G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='old_label')
        largest_cc = sorted(max(nx.connected_components(G_full), key=len))
        G = G_full.subgraph(largest_cc)
        G = nx.convert_node_labels_to_integers(G)   
        
    elif whichgraph == 'powerlaw':
        tpe = 'graph'
        G = nx.powerlaw_cluster_graph(params['n'], params['m'], params['p'])

    elif whichgraph == 'geometric':
        tpe = 'graph'
        G = nx.random_geometric_graph(params['n'], params['p'])
            
    elif whichgraph == 'powergrid':
        tpe = 'graph'
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

    elif whichgraph == 'S':   
        tpe = 'pointcloud'
        pos, color = skd.samples_generator.make_s_curve(params['n'], random_state=params['seed'])
        for i, _pos in enumerate(pos):
            G.add_node(i, pos = _pos, color = color[i])

    elif whichgraph == 'scale-free':
        tpe = 'graph'
        G = nx.DiGraph(nx.scale_free_graph(params['n']))                      
        
    elif whichgraph == 'SBM' or whichgraph == 'SBM_2':
        tpe = 'graph'
        G = nx.stochastic_block_model(params['sizes'],np.array(params['probs'])/params['sizes'][0], seed=params['seed'])
        for i,j in G.edges:
            G[i][j]['weight'] = 1.

        for u in G:
            G.nodes[u]['color'] = G.nodes[u]['block']

        G = nx.convert_node_labels_to_integers(G, label_attribute='labels_orig')        

    elif whichgraph == 'SM':
        tpe = 'graph'
        G = nx.newman_watts_strogatz_graph(params['n'], params['k'], params['p'])    
        
    elif whichgraph == 'swiss-roll':
        tpe = 'pointcloud'
        pos, color = skd.make_swiss_roll(n_samples=params['n'], noise=params['noise'], random_state=params['seed'])    
        for i, _pos in enumerate(pos):
            G.add_node(i, pos = _pos, color = color[i])
            
    elif whichgraph == 'torus':
        tpe = 'graph'
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=True)

        pos = {}
        for i in G:
            pos[i] = np.array(G.nodes[i]['old_label'])

    elif whichgraph == 'tree':
        tpe = 'graph'
        G = nx.balanced_tree(params['r'], params['h'])      
        
    elif whichgraph == 'tutte':
        tpe = 'graph'
        G = nx.tutte_graph()  
    else:
        raise Exception('Unknwon graph type, it will not work!')
    G.graph['name'] = whichgraph
    
    return G, tpe

# =============================================================================
# plot graph
# =============================================================================

def plot_graph_3D(G, node_colors='custom', edge_colors=[], params=None):
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
 
#    xyz = list(nx.get_node_attributes(G,'pos').values())   
    pos = nx.get_node_attributes(G, 'pos')       
    xyz = []
    for i in range(len(pos)):
        xyz.append(pos[i])
        
    xyz = np.array(xyz)
        
    #node colors
    if node_colors=='degree':
        edge_max = max([G.degree(i) for i in range(n)])
        node_colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)] 
    elif node_colors=='custom':
        node_colors = nx.get_node_attributes(G, 'color')
        node_colors = np.array([node_colors[i] for i in range(n)])  
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
            x = np.array((xyz[j[0]][0], xyz[j[1]][0]))
            y = np.array((xyz[j[0]][1], xyz[j[1]][1]))
            z = np.array((xyz[j[0]][2], xyz[j[1]][2]))
                   
            ax.plot(x, y, z, c=edge_color[i], alpha=0.5, linewidth = width[i])
    
    if edge_colors!=[]:    
        fig.colorbar(cmap)   
        
    if params==None:
        params = {'elev': 10, 'azim':290}
    elif params!=None and 'elev' not in params.keys():
        params['elev'] = 10
        params['azim'] = 290    
    ax.view_init(elev = params['elev'], azim=params['azim'])

    ax.set_axis_off()
 
    return fig       


def plot_graph(G, node_colors='cluster'):
        
    pos = list(nx.get_node_attributes(G,'pos').values())

    if node_colors=='cluster' and 'block' in G.nodes[1]:
        _labels = list(nx.get_node_attributes(G,'block').values())
    else:
        _labels = [0] * G.number_of_nodes()

    fig = plt.figure(figsize = (5,4))
    nx.draw_networkx_nodes(G, pos=pos, node_size=20, node_color=_labels, cmap=plt.get_cmap("tab20"))
    nx.draw_networkx_edges(G, pos=pos, width=1)

    plt.axis('off')
    
    return fig
