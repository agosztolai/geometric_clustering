# =============================================================================
# Input graphs
# =============================================================================
def inputGraphs(n):
    import networkx as nx
    import numpy as np
  
    if n == 1: #Watts-Strogatz      
        N = 20
        G = nx.newman_watts_strogatz_graph(N, 2, 0.30)  
        for i, j in G.edges():
            G[i][j]['weight'] = 1.
    
        x = np.linspace(0,2*np.pi,N)
        posx = np.cos(x)
        posy = np.sin(x)
        
        pos = []
        for i in range(N):
            pos.append([posx[i],posy[i]])
            
    elif n == 2: #Symmetric barbell graph      
        N = 10
        A = np.vstack((np.hstack((np.ones([N//2,N//2]), np.zeros([N//2,N//2]))), np.hstack((np.zeros([N//2,N//2]), np.ones([N//2,N//2])))))
        A = A-np.eye(N)
        A[N//2-1,N//2] = 1; A[N//2,N//2-1] = 1
        G = nx.Graph(A)
        pos = nx.spring_layout(G)
        
    elif n == 3: #Triangle of triangles
        m = 1
        N = 3
        A = np.ones([N, N])-np.eye(N)
        A = np.kron(np.eye(N**m),A)
        A[2,3]=1; A[3,2]=1; A[1,6]=1; A[6,1]=1; A[4,8]=1; A[8,4]=1
        A = np.vstack((np.hstack((A, np.zeros([9, 9]))), np.hstack((np.zeros([9, 9]), A))))
        A[0,9]=1; A[9,0]=1
    
        G = nx.Graph(A); 
        pos = nx.spring_layout(G,iterations=1000)
        
    elif n == 4: #Ring of cliques      
        num_cliques = 5
        clique_size = 6
        G = nx.ring_of_cliques(num_cliques, clique_size)
        
        x1 = np.linspace(-np.pi,np.pi,num_cliques)
        x2 = np.linspace(0,2*np.pi,clique_size)[::-1]
               
        posx = np.zeros(num_cliques*clique_size)
        posy = np.zeros(num_cliques*clique_size)
        for i in range(num_cliques):         
            for j in range(clique_size):
                posx[i*clique_size + j] = np.cos(x1[i]) + 0.5*np.cos(x2[j] + x1[i] + 2*np.pi*3/5)
                posy[i*clique_size + j] = np.sin(x1[i]) + 0.5*np.sin(x2[j] + x1[i] + 2*np.pi*3/5)
                
        pos = []
        for i in range(num_cliques*clique_size):
            pos.append([posx[i],posy[i]])
#        pos = nx.spring_layout(G,iterations=1000)
        
    elif n == 5: #SBM
        k = 3
        sizes = [75, 75, 300]
        
#        probs = 0.2*np.ones([k,k]); # Erdos-Renyi G(N,0.2)
        probs = np.ones([k,k]) + 0.4*np.diag(np.ones([1,k])) #communities same densities
#        M = 0.01*np.ones([k,k]) + np.diag([0.2, 0.4, 0.8]) #communities with different densities 
#        probs = 0.05*np.ones([k,k]) + 0.5*np.diag(np.ones([1,k])); # assortative dynamics
#        probs = 0.05*np.ones([k,k]) + 0.5*np.diag(np.ones([1,k])) + ...
#         0.1*np.diag(np.ones(1,k-1),1) + 0.1*np.diag(np.ones(1,k-1),-1); % ordered communities
              
        G = nx.stochastic_block_model(sizes, probs, seed=0)
        pos = nx.spring_layout(G,iterations=1000)
        
    elif n == 6: #Girvan-Newman
        k = 4
        g = 32
        p_in = 0.45
        p_out = (0.5-p_in)/3

        G = nx.planted_partition_graph(k, g, p_in, p_out, seed=0)
        pos = nx.spring_layout(G,iterations=100)
        
        truth = np.ones([1,g])
        for i in range(k-1):
            truth = np.concatenate((truth,(i+2)*np.ones([1,g])),axis = 1)
                   
    elif n == 7: #LFR
        N = 100 #number of nodes
        tau1 = 1.0# power law exponent of in degree
        tau2 = 1.0#power law exponent of community size
        mu = 0.5#fraction of edges per vertex to be shared with other comms
        
        G = nx.LFR_benchmark_graph(N, tau1, tau2, mu, average_degree=None, min_degree=None, tol=1e-07, max_iters=500, seed=0)
        pos = nx.spring_layout(G,iterations=1000)
        
#    elif n == 8: 
#        G = nx.karate_club_graph()
#        A = nx.adjacency_matrix(G)

    return G, pos