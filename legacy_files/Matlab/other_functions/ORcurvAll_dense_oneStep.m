function K = ORcurvAll_dense_oneStep(A,d,alpha)

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
all = 0;
epsi = 1e-6;
% loop over every edge once
if all == 1
    [x,y,~] = find(triu(ones(N))>0);
% x = 1:N; y = 1:N;
else
    [x,y,~] = find(triu(A)>0);
end

ind = (x-y)~=0;
x = x(ind); y = y(ind);

K = zeros(N);
for i = 1:length(x)

        % neighbourhoods x u Nx and y u Ny
        Nx = [x(i) find(A(x(i),:))]; 
        Ny = [y(i) find(A(y(i),:))];

        % diffusion distances between x u Nx and y u Ny
%         m = nnz(Nx); n = nnz(Ny);
        
        % distribution at x and y supported by the neighbourhood Nx and Ny
        mx = [alpha A(x(i),Nx(2:end))*(1-alpha)/sum(A(x(i),Nx(2:end)))];
        my = [alpha A(y(i),Ny(2:end))*(1-alpha)/sum(A(y(i),Ny(2:end)))];

        % Wasserstein distance between mx and my    
%         dNxNy = reshape(d(Nx,Ny),1,m*n);
        W = W1(mx,my,dNxNy);

        % curvature along x-y
        K(x(i),y(i)) = 1 - W/(d(x(i),y(i)) + epsi);

end

K = K + K';