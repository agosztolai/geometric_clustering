clear all; close all
addpath([pwd,'/functions']) % location of auxiliary functions

% parameters
alpha = 0.5; % lazy random walk parameter
whichgraph = 13; %choose a number as in inputGraphs.m
vis = 0; % plot visible

t = logspace(-2,3,20); % diffusion time scale 
l = 10; % number of retained eigenvalues (dimension of manifold)

% specify the graph or load some other graph
[G,A,X,Y] = inputGraphs(whichgraph); 
f = figure('Visible',vis);

%compute geodesic distances
d = distGeo(A);

v = 0;
for i = 1:length(t)
    disp(i)

    % compute curvatures for all edges
    [~, Phi] = distDiff(A,t(i),l);
%     K = ORcurvAll_dense_oneStep(A,d,alpha);
%     K = ORcurvAll_sparse_oneStep(A,d,alpha);
%     K = ORcurvAll_dense_full(d,Phi);
    K = ORcurvAll_sparse_full(A,d,Phi,0.99);
        
    % update edge weights and curvatures
    G.Edges.Weight = nonzeros(tril(A));
    indnonzeros = find(tril(A)); %edges with positive weights may have 0 kappa
    G.Edges.Kappa = K(indnonzeros);
 
    % append to movie
    frame = plotcurv(G,X,Y,f); 
    if i < length(t); stopmov = 0; else; stopmov = 1; end
    v = createMovie(frame, v, stopmov);      

end