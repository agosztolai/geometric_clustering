import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score as score
from pygenstability.pygenstability import WorkerVI

data = pkl.load(open('data/hox_gene_expression.pkl','rb'))[:-1]
c = ['C0', 'C1', 'C2']

plt.figure(figsize=(5, 2))
ax1 = plt.gca()
#plt.twinx()
#ax2 = plt.gca()

for i, method in enumerate(['geometric_modularity', 'markovstab']):
    cluster_results = pkl.load(open('jaccard/' + method + '_results.pkl','rb'))
    times = cluster_results['times']
    community_id = cluster_results['community_id']


    ground_truth_1 = np.unique(data['Neurotransmitter'].to_list(), return_inverse=True)[1]
    ground_truth_2 = np.unique(data['Neuron Class'].to_list(), return_inverse=True)[1]

    f_1, f_2 = [], []
    n_c = []
    for _i in range(len(times)):
        score = WorkerVI([community_id[_i], ground_truth_1])
        f_1.append(score([0, 1]))
        score = WorkerVI([community_id[_i], ground_truth_2])
        f_2.append(score([0, 1]))

        #f_1.append(score(community_id[_i], ground_truth_1))
        #f_2.append(score(community_id[_i], ground_truth_2))
        n_c.append(len(np.unique(community_id[_i])))
    #print(method, 'gt1 =', np.max(f_1), ' gt2 =', np.max(f_2))
    print(method, 'gt1 =', np.min(f_1), ' gt2 =', np.min(f_2))
    #ax2.plot(np.log10(times), n_c, '-+', c=c[i], label='number of clusters')
    #ax2.axhline(len(np.unique(ground_truth_1)), c='b', label='number of gt1 clusters')
    #ax2.axhline(len(np.unique(ground_truth_2)), c='g', label='number of gt2 clusters')
    #ax1.plot(np.log10(times), f_1, c=c[i], ls='-', label=method + ' gt1')
    ax1.plot( np.log10(times), f_2, c=c[i],  ls='-', label=method) # + ' gt2')
    plt.ylabel('VI')
    ax1.legend(loc='best')
    plt.axis([np.log10(times[0]), np.log10(times[-1]), 0, 0.85])
    #ax2.legend(loc='upper right')
plt.savefig('ground_truth_VI.svg')
plt.show()
