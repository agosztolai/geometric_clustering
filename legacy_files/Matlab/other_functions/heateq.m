%This script computes the heat equation on a graph.

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

Deg = full(diag(sum(A, 2))); %degree matrix
L = Deg - A; %Laplacian
[V, Lambda] = eig(L); %eigenvalues/vectors of L
Lambda = diag(Lambda);

%Initial condition 
Phi0 = zeros(N, N);
Phi0(2:5, 2:5) = 5; Phi0(10:15, 10:15) = 10; Phi0(2:5, 8:13) = 7;
Phi0 = Phi0(:);

Phi0V = V'*Phi0; %Transform IC into eigenvector coordinates 
for t = 0:0.05:5
   
%    Phi = Phi0V.*exp(-Lambda*t); %Exponential decay for each component
%    Phi = V*Phi; %Transform back to original coordinates
   Phi = expm(-t*L)*Phi0;
   Phi = reshape(Phi, N, N);
   
   %Plot
   imagesc(Phi);
   caxis([0, 10]);
   title(sprintf('Diffusion t = %3f', t));
   frame = getframe(1);
   im = frame2im(frame);
   [imind, cm] = rgb2ind(im, 256);
   
end