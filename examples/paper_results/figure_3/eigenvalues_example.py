import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from geocluster import curvature as cv
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
import scipy as sp
import time
import matplotlib.cm as cm

if __name__ == "__main__":

    tau = 0.5
    l = 2
    g = 4000
    seed = 1
    k_in, k_out = 4, 1 #this is in the sparse regime
      
    lambda2 = 2*k_out/(k_in + k_out)
    k = k_in + k_out
    rks = (k - np.sqrt(k))/(k + np.sqrt(k))
    
    # #eigenvalues dense regime
    # p_in = 0.5
    # p_out = 0.1
    # graph = nx.planted_partition_graph(l, 200, p_in, p_out, seed=seed)
    # largest_cc = max(nx.connected_components(graph), key=len)
    # graph = graph.subgraph(largest_cc)

    # laplacian = cv._construct_laplacian(graph,use_spectral_gap=False)
    # w, v = np.linalg.eig(laplacian.toarray())
            
    # plt.figure()
    # plt.hist(w,bins=50,density=True)
    # plt.axvline(2*p_out/(p_in + p_out),c='r')
    # # plt.savefig('dense_spectrum.svg')
    
    
    # #eigenvalues sparse regime
    # w_ensemble = []
    # for i in tqdm(range(5)):
    
    #     graph = nx.planted_partition_graph(l, g, k_in/g, k_out/g, seed=i)
    #     largest_cc = max(nx.connected_components(graph), key=len)
    #     graph = graph.subgraph(largest_cc)

    #     laplacian = cv._construct_laplacian(graph,use_spectral_gap=False)
    #     w, v = np.linalg.eig(laplacian.toarray())
    #     w_ensemble.append(w)
        
    # w_ensemble = np.hstack(w_ensemble)
    
    # plt.figure()
    # plt.hist(w_ensemble,bins=50,density=True)
    # plt.axvline(2*k_out/(k_in + k_out),c='r')
    # # plt.savefig('sparse_spectrum.svg')
    
    
    # #eigenvetors in sparse regime
    # graph = nx.planted_partition_graph(l, g, k_in/g, k_out/g, seed=seed)
    # largest_cc = max(nx.connected_components(graph), key=len)
    # graph = graph.subgraph(largest_cc)

    # laplacian = cv._construct_laplacian(graph,use_spectral_gap=False)
    # w, v = np.linalg.eig(laplacian.toarray())
    
    # #do sorting on eigenvectors
    
    # plt.figure()
    # for i in range(10):
    #     plt.plot(v[:,i])
    
    
    #when summing over eigenvectors in clusters and taking differences, the fluctuations vanish
    # g_range = np.arange(50,2000,500)
    g_range = [1000]
    
    sum_v_g = []
    for g_ in g_range:
        graph = nx.planted_partition_graph(l, g_, k_in/g_, k_out/g_, seed=seed)
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc)

        laplacian = cv._construct_laplacian(graph,use_spectral_gap=False)
        
        t = time.time()
        w, v = sp.linalg.eig(laplacian.toarray())
        # w,v = eigsh(laplacian.toarray(), k=500, which='SM')
        print(time.time() - t)
        
        C_1 = [i for i, k in enumerate(largest_cc) if k<g_]
        C_2 = [j for j, k in enumerate(largest_cc) if k>=g_]
        diffs = []
        for s in tqdm(range(v.shape[1])):
            diff = (v[C_1,s].sum() - v[C_2,s].sum())/np.sqrt(g_)
            diffs.append(diff)#np.exp(-w[s]/lambda2)*
    
        sum_v_g.append(diffs)
        
    all_g = [g_ for i, g_ in enumerate(g_range) for j in range(len(sum_v_g[i]))]
        
    plt.figure()
    plt.scatter(all_g,abs(np.hstack(sum_v_g)))
    plt.ylabel(r'$|\sum_{ij} (\phi_s(i) - \phi_s(j))\,\delta(C_i,C_j)|$')
    plt.xlabel('number of nodes')
    
    ind = np.where(abs(np.hstack(sum_v_g))==max(abs(np.hstack(sum_v_g))))[0]
    plt.figure()
    plt.plot(v[:,ind])
    plt.savefig('phi2.svg')
    
    plt.figure()
    ind = np.argsort(w)
    # colors = cm.rainbow(np.linspace(0, 1, len(w)))
    color = []
    for i, y in enumerate(abs(np.array(sum_v_g).flatten())):
        plt.scatter(w[i], y, color=cm.turbo(y))
        color.append(cm.turbo(y))
    # plt.scatter(w[ind],abs(np.array(sum_v_g)))
    plt.savefig('diffusion_differece.svg')
    #plt.axvline(lambda2,c='r')
    
    #compute correlation with ground truth
    gt = [1/np.sqrt(g_) if i in C_1 else -1/np.sqrt(g_) for i in range(len(largest_cc))]
    gt = np.array(gt)
    corr = []
    for s in range(v.shape[1]):
        corr.append(gt.dot(v[:,s]))
    
    plt.figure()
    for i, y in enumerate(corr):
        plt.scatter(w[i],y,color=color[i])
    plt.savefig('correlation.svg')
    
    
    # diffs = []
    # eigs = []
    # for g_ in tqdm(range(5)):
    #     graph = nx.planted_partition_graph(l, g, k_in/g, k_out/g, seed=seed)
    #     largest_cc = max(nx.connected_components(graph), key=len)
    #     graph = graph.subgraph(largest_cc)

    #     laplacian = cv._construct_laplacian(graph,use_spectral_gap=False)
    #     w, v = np.linalg.eig(laplacian.toarray())
        
    #     C_1 = [i for i, k in enumerate(largest_cc) if k<g_]
    #     C_2 = [j for j, k in enumerate(largest_cc) if k>=g_]
    #     mean_v = []
 
    #     for s in range(len(v)):     
    #         diffs.append((v[C_1,s].sum() - v[C_2,s].sum())/np.sqrt(len(largest_cc)))
    #         eigs.append(w[s])
    #         # mean_v.append(np.mean(diffs))
            
    #     # sum_v_g.append(np.array(mean_v))
    
    # plt.figure()
    # plt.scatter(w,abs(sum_v_g[-1]))
    # plt.axvline(lambda2,c='r')
    
    # g=1000
    # graph = nx.planted_partition_graph(l, g, k_in/g, k_out/g, seed=seed)
    # largest_cc = max(nx.connected_components(graph), key=len)
    # graph = graph.subgraph(largest_cc)

    # laplacian = cv._construct_laplacian(graph,use_spectral_gap=False)
    # _, v = np.linalg.eig(laplacian.toarray())
        
    # C_1 = [i for i, k in enumerate(largest_cc) if k<g]
    # C_2 = [j for j, k in enumerate(largest_cc) if k>=g]
        
    # evec_diff = []
    # pairs = [(x,y) for x in C_1 for y in C_2]
    # # for s in range(len(v)):
    # for pair in pairs:
    #     evec_diff.append(v[pair[0],5]- v[pair[1],5])
        
    # plt.hist(evec_diff,bins=50)
