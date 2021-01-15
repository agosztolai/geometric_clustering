def label_to_int(label):
    convert = {}
    int_labels = []
    for l in label:
        if l not in convert.keys():
            convert[l] = len(convert)
        int_labels.append(convert[l])
    return int_labels
    
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nMI
# from sklearn.metrics import normalized_mutual_info_score

cluster_results = pkl.load(open('jaccard/geometric_modularity_results.pkl','rb'))
times = cluster_results['times']
community_id = cluster_results['community_id']

data = pkl.load(open('data/hox_gene_expression.pkl','rb'))
ground_truth_1 = list(data['Neurotransmitter'][:-2])
ground_truth_1 = label_to_int(ground_truth_1)
ground_truth_2 = list(data['Neuron Class'][:-2])
ground_truth_2 = label_to_int(ground_truth_2)

f_1, f_2 = [], []
for i in range(len(times)):
    f_1.append(nMI(community_id[i], ground_truth_1))
    f_2.append(nMI(community_id[i], ground_truth_2))
    
plt.plot(np.log10(times), f_1, np.log10(times), f_2)
plt.savefig('mi.svg')