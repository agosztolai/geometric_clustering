function [kappa, path] = ORcurvxy_geo(G,x,y,alpha)
%Ollivier-Ricci curvature between two prob. measures mx(u) and my(v), 
%supported on x and Nx = {u: u~x} and y and Ny = {v: v~y}, respectively. 
%mx and my are the one-step transition probabilities of a lazy random walk, 
%defined as mx(u) = alpha if u=x and (1-alpha)/dx if u~x and 0 otherwise.
%The distance is measured on the graph G=(V,E,w), which is a Matlab object.

% neighbourhoods of x and y
Nx = [x; neighbors(G,x)];
Ny = [y; neighbors(G,y)];

% all shortest paths between Nx and Ny
m = length(Nx); n = length(Ny);
d = zeros(m,n);
for i = 1:m
    for j = 1:n
        [~,dij] = shortestpath(G,Nx(i),Ny(j));
        d(i,j) = dij;
    end
end

d = reshape(sparse(d'),1,m*n);

% distribution at x and y supported by the neighbourhood Nx and Ny
mx = zeros(1,m); mx(1) = alpha; mx(2:m) = (1-alpha)/(m-1);
my = zeros(1,n); my(1) = alpha; my(2:n) = (1-alpha)/(n-1);

% Wasserstein distance between mx and my
W = W1(mx,my,d);

% geodesic distance between x and y
[path,dxy] = shortestpath(G,x,y);

% curvature along x-y
kappa = 1 - W/dxy;