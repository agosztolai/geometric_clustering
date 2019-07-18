function K = ORcurvAll_dense_full(d,Phi)

%Ollivier-Ricci curvature between two prob. measures mx(u) and my(v). 
%mx and my are distributions given by two diffusion processes with initial
%conditions \delta_x and \delta_y after time t (implicit in Phi)

%INPUT: A adjacency matrix
%       d distance matrix (pairwise geodesic or diffusion distances)
%       Phi = exp(-tL)

%OUTPUT: K NxN matrix with entries kij marking the curvature between
%nodes i and j

N = size(d,1);
% loop over every edge once
[x,y,~] = find(triu(ones(N))>0);
% x = 1:N; y = 1:N;
ind = (x-y)~=0;
x = x(ind); y = y(ind);

K = zeros(N);
for i = 1:length(x)
        
        % distribution at x and y supported by the neighbourhood Nx and Ny
        mx = Phi(x(i),:); mx(mx<eps) = 0;
        my = Phi(y(i),:); my(my<eps) = 0;

        % Wasserstein distance between mx and my    
%         dNxNy = reshape(d',1,N*N);
        W = W1(mx,my,dNxNy);

        % curvature along x-y
        K(x(i),y(i)) = 1 - W/d(x(i),y(i));

end

K = K + K';