function [d, Phi] = distDiff(A,t,l,dist)

if nargin < 4; dist = 0; d = 0; end
if nargin < 3; dist = 0; l = 0; d = 0; end

N = size(A,1); %number of nodes
D = full(diag(sum(A,2))); %degree matrix

% L = D^(-1/2)*A*D^(-1/2); %random walk Laplacian
L = D - A; %combinatorial Laplacian
L = D^(-1/2)*L*D^(-1/2); %normalised laplacian

if l == 0
    Phi = expm(-t*L);
else %this is slower
    [V,Lambda] = eig(L);
    Lambda = diag(Lambda);
    Phi = V'.*exp(-Lambda*t); %diffusion map
    Phi = V*Phi;
end

Phi(Phi<eps) = 0;

%all diffusion distances
if dist == 1 
    d = distmat1(Phi);
    d(d<eps) = 0;
end