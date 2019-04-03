function [KappaL,KappaU] = ORcurvAll_sparse(A,d,Phi,cutoff,lambda)

%Ollivier-Ricci curvature between two prob. measures mi(k) and mj(l), which
%are defined as mi(k) = {Phi}ik, where Phi = Phi(t) = expm(-t*L).

%INPUT: A adjacency matrix
%       d distance matrix

%OUTPUT: KappaL, KappaU NxN matrices with entries kij marking the  lower
%        bound and upper bound on the OR curvature between nodes i and j

% loop over every edge once
N = size(A,1);
KappaU = zeros(N); KappaL = zeros(N);

[x,y,~] = find(triu(A)>0);
for i = 1:length(x)

    % distribution at x and y supported by the neighbourhood Nx and Ny
    mx = Phi(x(i),:); 
    my = Phi(y(i),:); 
    
    % Prune small masses to reduce problem size
    if cutoff < 1
        [smx,Nx] = sort(mx,'descend'); 
        [smy,Ny] = sort(my,'descend');
    
        cmx = cumsum(smx); Nx = Nx(logical([1 cmx(2:end) < cutoff])); 
        cmy = cumsum(smy); Ny = Ny(logical([1 cmy(2:end) < cutoff]));        
        
        % Restrict distance matrix 
        dNxNy = d(Nx,Ny);
        
        % renormalise 
        mx = mx(Nx)/sum(mx(Nx)); my = my(Ny)/sum(my(Ny));
    else
        dNxNy = d;
        mx(mx<eps) = 0; my(my<eps) = 0;
        
        % renormalise 
        mx = mx/sum(mx); my = my/sum(my);
    end
    
    % curvature along x-y
    if lambda < inf %approximate solution
        K = exp(-lambda*dNxNy);
        [U,L,~,~] = sinkhornTransport(mx',my',K,K.*dNxNy,lambda,[],[],[],[],0);
        KappaL(x(i),y(i)) = 1 - U/d(x(i),y(i));  
        KappaU(x(i),y(i)) = 1 - L/d(x(i),y(i));      
    else %exact solution
        W = W1(mx,my,dNxNy);
        KappaU(x(i),y(i)) = 1 - W/d(x(i),y(i));  
        KappaL = KappaU;
    end   
end

KappaU = KappaU + KappaU'; 
KappaL = KappaL + KappaL';
KappaU(abs(KappaU)<eps) = 0; 
KappaL(abs(KappaL)<eps) = 0;