function [G, A, X, Y] = inputGraphs(i)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% 1: finite grid (specify n)
% 2: some graph
% 3: 4-regular tree
% 4: 3D discrete cube with Hamming metric
% 5: n-clique (specify n)
% 6: soccer ball
% 7: 2 communities using SBM (specify n, seed, memberships)
% 8: 3 communities using SBM 
% 9: 3 ordered communities using SBM
%10: Grid
%11: Barbell graph
%12: Hierarchical (Ravasz-Barabasi)
%13: Hierarchical 

X = []; Y = [];
% Finite grid
if i==1
    
    n = 10;
    A = delsq(numgrid('S',n+2));
    A = -1*((A + A')/2 - diag(diag(A)));
    
    G = graph(A);

% some graph    
elseif i == 2   
    
    s = [1 1 1 2 2 6 6 7 7 3 3 9 9 4 4 11 11 8];
    t = [2 3 4 5 6 7 8 5 8 9 10 5 10 11 12 10 12 12];
    w = [10 10 10 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1];
    G = graph(s,t,w); %digraph() if directed

% 4-regular tree    
elseif i == 3  
    
    s = [1 2 2 2 3 3 3 4  4  4  5  5  5  1  1];
    t = [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16];
    w = [1 1 1 1 1 1 1 1  1  1  1  1  1  1  1];
    G = graph(s,t,w);
  
% 3D discrete cube with Hamming metric    
elseif i == 4     
    
    s = [1 1 1 2 2 3 3 4 5 5 6 7];
    t = [2 4 8 3 7 4 6 5 6 8 7 8];
    w = [1 1 1 1 1 1 1 1 1 1 1 1];
    G = graph(s,t,w);

% n-clique
elseif i == 5
    
    n = 5;
    A = ones(n)-diag(ones(n,1));
    G = graph(A);

% soccer ball
elseif i == 6
    
    G = graph(bucky);

% Two communities using SBM
elseif i == 7
    
    n = 500; k = 2;
    r = 1:n;
    c = 1*(r<ceil(n/2)) + 2*(r>=ceil(n/2));
    M = 0.01*ones(k,k) + 0.4*diag(ones(1,k));
    A = sbm(c,M);
    G = graph(A);
    
% Three communities using SBM with different sizes and densities    
elseif i == 8
    
    n = 70; k = 3;
    r = 1:n;
    c = 1*(r<40) + 2*((r<60)&(r>=40)) + 3*(r>=60);
    M = 0.01*ones(k,k) + diag([0.2 0.4 0.8]);
    A = sbm(c,M);
    G = graph(A);
    
% 3 ordered communities using SBM    
elseif i == 9
    n = 100;
    r = 1:n;
    c = 1*(r<=33) + 2*((33<r)&(r<=66)) + 3*(r>66);
    k = 3; % community size

%     M = 0.2*ones(k,k); % erdos-renyi G(n,p)
%     M = 0.05*ones(k,k) + 0.5*diag(ones(1,k)); % assortative dynamics
    M = 0.05*ones(k,k) + 0.5*diag(ones(1,k)) + ...
        0.1*diag(ones(1,k-1),1) + 0.1*diag(ones(1,k-1),-1); % ordered communities
    A = sbm(c,M);
    G = graph(A);
    
% Grid    
elseif i == 10
    
N = 20;%The number of pixels along a dimension of the image
A = zeros(N*N, N*N);%The adjacency matrix

%Use 8 neighbors, and fill in the adjacency matrix
%\  |  /
%- i,j -
%/  |  \
dx = [-1, 0, 1, -1, 1, -1, 0, 1];
dy = [-1, -1, -1, 0, 0, 1, 1, 1];
for x = 1:N
   for y = 1:N
       index = (x-1)*N + y;
       for ne = 1:length(dx)
           newx = x + dx(ne);
           newy = y + dy(ne);
           if newx > 0 && newx <= N && newy > 0 && newy <= N
               index2 = (newx-1)*N + newy;
               A(index, index2) = 1;
           end
       end
   end
end
G = graph(A);

% Symmetric barbell graph
elseif i == 11
    
    n = 10;
    A = [ones(n/2) zeros(n/2); zeros(n/2) ones(n/2)];
    A = A-eye(n);
    A(n/2,n/2+1) = 1; A(n/2+1,n/2) = 1;
    G = graph(A);

% Hierarchical (Ravasz-Barabasi) graph    
elseif i == 12
    m = 2;
    n = 5;
    A = ones(n)-eye(n)-fliplr(eye(n));
    A(3,3) = 0;
    X = [1 2 1.5 1 2];
    Y = [1 1 1.5 2 2];
    
    A = kron(eye(n^m),A);
    A(:,ceil(n^(m+1)/2)) = repmat([1 1 0 1 1]', n^m, 1);
    A(ceil(n^(m+1)/2),:) = repmat([1 1 0 1 1]', n^m, 1);
    G = graph(A);
    Z = zeros(1,n^(m+1));
    del = [ 0 0 0 1 -1; -1 1 0 0  0];
    
    for i = 1:m
        del = 1.6^i*kron(del,ones(1,n));
        del = (del'*[1 -1; 1 1]^(mod(i,2)==0))';
        X = repmat(X,1,n) + del(1,:);
        Y = repmat(Y,1,n) + del(2,:); 
    end
    
elseif i == 13
    m = 1;
    n = 3;
    A = ones(n)-eye(n);
    A = kron(eye(n^m),A);
    A(3,4)=1; A(4,3)=1; A(2,7)=1; A(7,2)=1; A(5,9)=1; A(9,5)=1;
    A = [A zeros(9); zeros(9) A];
    A(1,10) =1; A(10,1)=1;
    
    G = graph(A);  
end

% reconstruct weighted adjacency matrix
if ~exist('A','var')
    N = numnodes(G);
    [s,t] = findedge(G);
    A = sparse(s,t,G.Edges.Weight,N,N); % directed graph
    A = A + A.' - diag(diag(A)); % undirected graph
end

A = sparse(A);

end

function W = sbm(c,P,directed)

%   generateSbm(c,P) generates a graph adjacency matrix using a stochastic
%   block model where the class membership of each node is given by the
%   vector c, and the matrix P contains the probability of forming edges
%   between each pair of classes.
%
%   generateSbm(c,P,directed) allows for the creation of directed graphs by
%   setting directed to true. By default, undirected graphs are created.

% Author: Kevin S. Xu

% Set as undirected graph by default
if nargin < 3, directed = false; end

n = length(c);
k = max(c);

if (size(c,1) ~= 1) && (size(c,2) ~= 1)
    error('c must be a vector')
end
c = reshape(c,n,1);

assert(size(P,1)>=k,'P must be a k-by-k matrix, where k >= max(c)')
if directed == false
    assert(isequal(P,P'),'P must be a symmetric matrix')
end

W = zeros(n,n); % Graph adjacency matrix
for c1 = 1:k
    % First consider connections between two nodes in the same class
    inC1 = (c==c1);
    numC1 = sum(inC1);
    
    % Form edges between nodes in same class only if more than one node in
    % the class
    if numC1 > 1
        if directed == true
            W_Block = zeros(numC1,numC1);
            blockMask = ~diag(true(numC1,1));
            W_Block(blockMask) = bernrnd(P(c1,c1),numC1*(numC1-1),1);
            W(inC1,inC1) = W_Block;
        else
            W(inC1,inC1) = squareform(bernrnd(P(c1,c1), ...
                numC1*(numC1-1)/2,1));
        end
    end
    
    % Form edges between nodes in different classes. Loop start index
    % depends on whether graph is directed or undirected.
    if directed == true
        start = 1;
    else
        start = c1+1;
    end
    for c2 = start:k
        % Diagonal blocks are already considered, so ignore them in the
        % loop
        if c2 == c1
            continue
        end
        
        inC2 = (c==c2);
        numC2 = sum(inC2);
        if numC2 == 0
            continue
        end
        W(inC1,inC2) = bernrnd(P(c1,c2),numC1,numC2);
        if directed == false
            W(inC2,inC1) = W(inC1,inC2)';
        end
    end
end
end

function x = bernrnd(p,m,n)
% bernrnd(p,m,n) generates an m-by-n matrix of realizations of Bernoulli 
% random variables with parameter p.

if nargin < 2
    m = 1;
end
if nargin < 3
    n = m;
end

assert((p>=0) && (p<=1), ['p must be between 0 and 1. Currently p = ' ...
    num2str(p)])
x = binornd(1,p,m,n);

% % Implementation not requiring Statistics Toolbox
% y = rand(m,n);	% Uniformly distributed random variables
% x = zeros(m,n);
% x(y < p) = 1;
end