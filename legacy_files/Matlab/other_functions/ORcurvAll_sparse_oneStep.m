function K = ORcurvAll_sparse_oneStep(A,d,alpha)

%Ollivier-Ricci curvature between two prob. measures mx(u) and my(v), 
%supported on x and Nx = {u: u~x} and y and Ny = {v: v~y}, respectively. 
%mx and my are the one-step transition probabilities of a lazy random walk, 
%defined as mx(u) = alpha if u=x and (1-alpha)/dx if u~x and 0 otherwise.

%INPUT: A adjacency matrix
%       d distance matrix
%       alpha parameter for lazy random walk

%OUTPUT: K NxN matrix with entries kij marking the curvature between
%nodes i and j

N = size(A,1);
% loop over every edge once
[x,y,~] = find(triu(A)>0);
K = zeros(N);
for i = 1:length(x)

        % neighbourhoods x u Nx and y u Ny
        Nx = [x(i) find(A(x(i),:))]; %A(x(i),:)>0 + sparse(1,x(i),1,1,N);%
        Ny = [y(i) find(A(y(i),:))]; %A(y(i),:)>0 + sparse(1,y(i),1,1,N);%

        % diffusion distances between x u Nx and y u Ny
%         m = length(Nx>0); n = length(Ny>0); 
        m = nnz(Nx); n = nnz(Ny);
        dNxNy = reshape(d(Nx,Ny),1,m*n);
        
        % distribution at x and y supported by the neighbourhood Nx and Ny
%         mx = zeros(1,m); mx(1) = alpha; mx(2:m) = (1-alpha)/(m-1);
%         my = zeros(1,n); my(1) = alpha; my(2:n) = (1-alpha)/(n-1);
        mx = zeros(1,m); mx(1) = alpha; mx(2:m) = A(x(i),Nx(2:end))*(1-alpha)/sum(A(x(i),Nx(2:end)));
        my = zeros(1,n); my(1) = alpha; my(2:n) = A(y(i),Ny(2:end))*(1-alpha)/sum(A(y(i),Ny(2:end)));

        % Wasserstein distance between mx and my    
        W = W1(mx,my,dNxNy);

        % curvature along x-y
        K(x(i),y(i)) = 1 - W/d(x(i),y(i));

end

K = K + K';