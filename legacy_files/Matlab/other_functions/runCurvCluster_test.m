% clear all, close all
addpath([pwd,'/functions']) % location of auxiliary functions

%% parameters
alpha = 0.0; % lazy random walk parameter
t = 0.0001; % diffusion time scale for ORcurvAll_diff only
l = 4; % accuracy (dimension of manifold) for ORcurvAll_diff only

%clustering strategy
strat = 1; k = 0; % 1: threshold (k), 2: Louvain, 3: greedy k = 12

%% load graph
if ~exist('G','var') 
    [G,A,X,Y] = inputGraphs(13); %graph
end

%% compute the Ollivier-Ricci curvature for all edges

% Distance matrix (uncomment as appropriate)
d = distGeo(A);
% d = distDiff(A,t,l);
% d = distRes(A,t,l);

% OR curvature
K = ORcurvAll_sparse(A,d,alpha);
% K = exp(K);%./sum(exp(K),2);
    
% update edge weights and curvatures
G.Edges.Weight = nonzeros(tril(A));
indnonzeros = find(tril(A)); %edges with positive weights may have 0 kappa
G.Edges.Kappa = K(indnonzeros);

%% cluster

%strategy 1 --> cut edges with curvature lower than KappaMin, then find
%connected components. 
if strat == 1
    %histogram of edge curvatures
%     histogram(G.Edges.Kappa,50) 
    
    %remove edges with small curvature
    ind = find(G.Edges.Kappa < k);
    G1 = rmedge(G,ind); 
    
    %find connected components
    bins = conncomp(G1); 
    
%strategy 3 --> Louvain    
elseif strat == 2

    bins = louvain(K);
     
%strategy 3 --> do DFS from a random vertex for highest curvature edges
%this is better then strategy 1 because it may detect more communities
elseif strat == 3
    
    bins = greedymax(K,k,p);
    
end

%% plot
plotcluster(G,1,bins,X,Y)