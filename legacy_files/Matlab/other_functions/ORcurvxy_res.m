function kappa = ORcurvxy_res(G,x,y,alpha,t)
%Ollivier-Ricci curvature between two prob. measures mx(u) and my(v), 
%supported on x and Nx = {u: u~x} and y and Ny = {v: v~y}, respectively. 
%mx and my are the one-step transition probabilities of a lazy random walk, 
%defined as mx(u) = alpha if u=x and (1-alpha)/dx if u~x and 0 otherwise.
%The distance is measured on the graph G=(V,E,w), which is a Matlab object.

%neighbourhoods x u Nx and y u Ny
Nx = [x; neighbors(G,x)];
Ny = [y; neighbors(G,y)];

W = adjacency(G); %weighted adjacency matrix
D = diag(sum(W,1)); %degree matrix
L = full(D-W);
Li = pinv(L);

%all shortest paths between Nx and Ny
m = length(Nx); n = length(Ny);
d = zeros(m,n);
vol = sum(sum(W))/2;
for i = 1:m
    for j = 1:n
        d(i,j) = vol*(Li(Nx(i),Nx(i)) + Li(Ny(j),Ny(j)) - 2*Li(Nx(i),Ny(j)));
    end
end

d = reshape(sparse(d'),1,m*n);

% distribution at x and y supported by the neighbourhood Nx and Ny
mx = zeros(1,m); mx(1) = alpha; mx(2:m) = (1-alpha)/(m-1);
my = zeros(1,n); my(1) = alpha; my(2:n) = (1-alpha)/(n-1);

% Wasserstein distance between mx and my
W = W1(mx,my,d);

% curvature along x-y
kappa = 1 - W/d(1,1);