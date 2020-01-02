import numpy as np
import scipy as sc
import scipy.io
import pylab as plt
import networkx as nx

def create_celegans(location):
    # load data
    data = scipy.io.loadmat(location+'celegans.mat')
    A = data['A_combined']
    G = nx.DiGraph(A)

    #set labels
    labels = {}
    for i in G:
        G.nodes[i]['labels'] = data['Labels'][i][0][0]
        labels[i] =  data['Labels'][i][0][0] 


    #set neuron types
    neuron_tpe = np.loadtxt(location+'neuron_type.txt',dtype=str)
    neuron_type = {}
    colors = []
    for i in range(len(neuron_tpe)):
        for il, l in enumerate(labels):
            if neuron_tpe[i][1] == labels[il]:
                if neuron_tpe[i][2] == 'M':
                    G.nodes[il]['type'] = 0
                    neuron_type[il] = 0 
                    colors.append('C0')
                    
                elif neuron_tpe[i][2] == 'I':
                    G.nodes[il]['type'] = 1   
                    neuron_type[il] = 1 
                    colors.append('C1')

                elif neuron_tpe[i][2] == 'S':
                    G.nodes[il]['type'] = 2
                    neuron_type[il] = 2 
                    colors.append('C2')


    #set positions of nodes, following SI of Varshney et al 2011
    A = np.array(nx.to_numpy_matrix(G))

    W = 0.5*(A + A.T)

    D = np.diag(np.array(W.sum(1)).flatten())
    b = np.array((W*np.sign(A-A.T)).sum(0)).flatten()

    L = D - W

    L = (nx.laplacian_matrix(nx.Graph(W))).toarray()

    z = np.array(np.linalg.pinv(L).dot(b)).flatten()

    #z = sc.sparse.linalg.spsolve(L,b)

    L_norm = nx.normalized_laplacian_matrix(nx.Graph(W))

    vs = sc.sparse.linalg.eigs(L_norm, which='SM', k=3)[1][:, 1:]
    D_sqrt_inv = np.diag(1/np.sqrt(np.diag(D)))

    v2 = D_sqrt_inv.dot(vs[:,0])
    v3 = D_sqrt_inv.dot(vs[:,1])
    pos = {}
    for i in G:
        #pos[i] = (-np.real(v2[i]), np.real(v3[i]))
        pos[i] = (-np.real(v2[i]), -z[i])
    
    return G, pos, labels, neuron_type, colors

def plot_celegans(G, pos, labels, color, plot_labels = False):
    plt.figure()
    nx.draw_networkx_nodes(G, pos = pos, node_size=100, node_color = color)
    nx.draw_networkx_edges(G, pos = pos, width = 1, alpha = 0.2)
    if plot_labels:
        nx.draw_networkx_labels(G, pos = pos, labels= labels, font_size = 5 )#node_color = np.array(list(neuron_type.values())))

    #plt.show()


