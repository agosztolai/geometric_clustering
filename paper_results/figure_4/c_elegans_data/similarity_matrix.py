import pickle as pkl
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from RMST import RMST

#metrics = ['dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 
#           'kulsinski', 'minkowski', 'rogerstanimoto', 
#           'russellrao', 'sokalmichener', 'sokalsneath', 
#           'yule']

metrics = ['jaccard']
eps = 1

data = pkl.load(open('data/hox_gene_expression.pkl','rb'))
exp = data.iloc[1:-1,4:].to_numpy()

#remove genes that can be found in almost every neuron or no neuron
id1 = exp.sum(0)<exp.shape[0]*0.9
id2 = exp.sum(0)>0

id3 = id1*id2
exp = exp[:, id3]

def label_to_int(label):
    convert = {}
    int_labels = []
    for l in label:
        if l not in convert.keys():
            convert[l] = len(convert)
        int_labels.append(convert[l])
    return int_labels

ground_truth_1 = list(data['Neurotransmitter'][:-2])
#ground_truth_1 = label_to_int(ground_truth_1)
ground_truth_1 = { i:j for i,j in enumerate(ground_truth_1)}
ground_truth_2 = list(data['Neuron Class'][:-2])
#ground_truth_2 = label_to_int(ground_truth_2)
ground_truth_2 = { i:j for i,j in enumerate(ground_truth_2)}
neuron = list(data.iloc[:-1,1])


for m in metrics:
    print(m)
    
    dist = pdist(exp, metric=m)
#    dist = 1.0/dist#
    dist = (max(dist)-dist)/max(dist)
#    dist = np.exp(-dist/eps)
    # dist[dist<0.19] = 0
    
    # pca = PCA(n_components=10)
    # principalComponents = pca.fit_transform(squareform(dist))
    # dist = pdist(principalComponents, metric='euclidean')
    # dist = (max(dist)-dist)/max(dist)
    # dist = np.exp(-dist/eps)
#    dist[dist<0.13] = 0
    # print(sum(pca.explained_variance_ratio_))
#    print(pca.singular_values_)
    
    G = nx.from_numpy_matrix(squareform(dist))
    
    G_RMST = RMST(G,gamma=0.01, weighted=True)
    # G_RMST = G
    
    for i in G_RMST:
        G_RMST.nodes[i]['neurotransmitter'] = ground_truth_1[i]
        G_RMST.nodes[i]['neuron_class'] = ground_truth_2[i]
        G_RMST.nodes[i]['neuron'] = neuron[i]
    
#    th=1
#    connected = False
#    while (~connected) & (th>=0):
#        dist_trun = dist.copy()
#        th -= 0.1
#        if th<0:
#            th=0
#        print(th)
#        dist_trun[dist_trun<max(dist)*th] = 0
#        dist_trun[dist_trun>=max(dist)*th] = 1
#    
#        G = nx.from_numpy_matrix(squareform(dist_trun))
#        
#        connected = nx.is_connected(G)
        
    assert nx.is_connected(G_RMST), 'graph is not connected'
        
#    labels = data.iloc[:-1,:4]
#    neuron_class = list(data.iloc[:-1,0])
#    neuron = list(data.iloc[:-1,1])
#    neurotransmitter = list(data.iloc[:-1,2])
#    neuron_type = list(data.iloc[:-1,3])
#    nx.set_node_attributes(G_RMST, neuron_class, name='neuron_class')
#    nx.set_node_attributes(G_RMST, neuron, name='neuron')
#    nx.set_node_attributes(G_RMST, neurotransmitter, name='neuron')
#    nx.set_node_attributes(G_RMST, neuron_type, name='neuron_type')
    nx.write_gpickle(G_RMST, "data/hox_gene_expression_" + m + ".gpickle", protocol=4)

#    data = pkl.load(open('data/all_gene_expression.pkl','rb'))
#    exp = data.iloc[1:,3:].to_numpy()
#    dist = pdist(exp, metric=m)
#    dist = np.exp(-dist**2/eps)
#    
#    th=max(dist)*0.1
#    connected = False
#    while (~connected) & (th>0):
#        dist_trun = dist.copy()
#        th -=0.01
#        if th<0:
#            th=0
#        print(th)
#        dist_trun[dist_trun<th] = 0
#    
#        G = nx.from_numpy_matrix(squareform(dist_trun))
#        
#        connected = nx.is_connected(G)
#    #genes = list(data.columns)[3:]
#    labels = data.iloc[:,:3]
#    neuron_class = list(data.iloc[:,0])
#    neuron = list(data.iloc[:,1])
#    neuron_type = list(data.iloc[:,2])
#    nx.set_node_attributes(G, neuron_class, name='neuron_class')
#    nx.set_node_attributes(G, neuron, name='neuron')
#    nx.set_node_attributes(G, neuron_type, name='neuron_type')
#    nx.write_gpickle(G, "data/all_gene_expression_" + m + ".gpickle")