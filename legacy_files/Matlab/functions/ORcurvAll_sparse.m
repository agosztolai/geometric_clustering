function [KappaL,KappaU] = ORcurvAll_sparse(E,d,Phi,cutoff,lambda)

%Ollivier-Ricci curvature between two prob. measures mi(k) and mj(l), which
%are defined as mi(k) = {Phi}ik, where Phi = Phi(t) = expm(-t*L).

%INPUT: E list of edges
%       d distance matrix

%OUTPUT: KappaL, KappaU NxN matrices with entries kij marking the  lower
%        bound and upper bound on the OR curvature between nodes i and j

% loop over every edge once
N = size(Phi,1);
KappaU = zeros(N); KappaL = zeros(N);

for i = 1:size(E,1)

    % distribution at x and y supported by the neighbourhood Nx and Ny
    mx = Phi(E(i,1),:); 
    my = Phi(E(i,2),:); 
    
%     if cutoffType == 1
%         [smx,Nx] = sort(mx,'descend'); 
%         [smy,Ny] = sort(my,'descend');    
%         cmx = cumsum(smx); Nx = Nx(logical([1 cmx(2:end) < cutoff])); 
%         cmy = cumsum(smy); Ny = Ny(logical([1 cmy(2:end) < cutoff]));                        
%     elseif cutoffType == 2
        Nx = mx>(1-cutoff)*max(mx); Ny = my>(1-cutoff)*max(my);      
%     end
    
    % restrict & renormalise 
    dNxNy = d(Nx,Ny); % Restrict distance matrix
    mx = mx(Nx); my = my(Ny);
    mx = mx/sum(mx); my = my/sum(my);
    
    % curvature along x-y
    if lambda < inf %approximate solution
        K = exp(-lambda*dNxNy);
        [U,L,~,~] = sinkhornTransport(mx',my',K,K.*dNxNy,lambda,[],[],[],[],0);
        KappaL(E(i,1),E(i,2)) = 1 - U/d(E(i,1),E(i,2));  
        KappaU(E(i,1),E(i,2)) = 1 - L/d(E(i,1),E(i,2));      
    else %exact solution
        W = W1(mx,my,dNxNy);
        KappaU(E(i,1),E(i,2)) = 1 - W/d(E(i,1),E(i,2));  
        KappaL = KappaU;
    end   
end

KappaU = KappaU + KappaU'; 
KappaL = KappaL + KappaL';
KappaU(abs(KappaU)<eps) = 0; 
KappaL(abs(KappaL)<eps) = 0;