clear all, close all
addpath([pwd,'/functions']) % location of auxiliary functions
%% computes the Ollivier-Ricci curvature between pair (x,y)

alpha = 0.0; % lazy random walk parameter
x=1; y=2; % label of vertices between which curvature is to be computed

t = 1; % diffusion time scale for ORcurvxy_diff only
l = 10; % accuracy (dimension of manifold) for ORcurvxy_diff only

[G,Aold] = inputGraphs(4); %graph

% uncomment as appropriate
[kxy, path] = ORcurvxy_geo(G,x,y,alpha);
% kxy = ORcurvxy_res(G,x,y,alpha,t);
% kxy = ORcurvxy_diff(G,x,y,alpha,t,l);

plotnhood(G,x,y,path) %plot and highlight nodes and shortest path