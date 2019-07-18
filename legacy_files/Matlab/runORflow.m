clear all; close all
addpath([pwd,'/functions']) % location of auxiliary functions

% d/dt( d(x,y) ) = -(k(x,y)-knorm)d(x,y)

% parameters
alpha = 0; % lazy random walk parameter
h = 0.01; % Euler step
numStep = 300; % number of Euler steps
invisible = 1; %display plot?
whichgraph = 11; %choose a number as in inputGraphs.m
t = 0.1; % diffusion time scale 
l = 11; % number of retained eigenvalues 

% [G,Aold] = inputGraphs(whichgraph); % specify the graph or load some other graph% load('sbm_2comms')

load('A.mat')
Aold = A;
G = graph(A);

% Distance matrix (uncomment as appropriate)
d = distGeo(Aold);
% d = distDiff(A,t,l);
% d = distRes(A,t,l);

% Initial OR curvature
Kold = ORcurvAll_sparse(Aold,d,alpha);

% OR flow evolution

v = 0;
for i = 1:numStep
    disp(i)

    % OR curvature    
    volG = sum(sum(Aold));
    Knorm = sum(sum(Aold.*Kold))/volG;
    
    % Euler step
    A = Aold - h*(Kold-Knorm).*Aold;
    
    % compute curvatures for all edges
    d = distGeo(Aold);
    % d = distDiff(A,t,l);
    % d = distRes(A,t,l);
    K = ORcurvAll_sparse(Aold,d,alpha);
    
    % mean squared error
    mse(i) = norm(K-Kold);
        
    % update edge weights and curvatures
    G.Edges.Weight = nonzeros(tril(A));
    indnonzeros = find(tril(A)); %edges with positive weights may have 0 kappa
    G.Edges.Kappa = K(indnonzeros);
    
    
    % plot every k step
    k = 5; 
    if mod(i,k) == 2
        frame = plotORflow(G,mse,i,invisible);     
        if i < numStep-k; stopmov = 0; else; stopmov = 1; end
        v = createMovie(getframe, v, stopmov); % append to movie     
    end

    Aold = A; Kold = K;
end