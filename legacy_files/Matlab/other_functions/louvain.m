function VV = louvain(A)
% INPUT
% A:      Weighted adjacency matrix
%
% OUTPUT
% VV:     N-by-1 matrix, VV(n) is the cluster to which node n belongs 
%

s = 1; % s : 1 = Recursive computation 
       %     0 = Just one level

N = length(A);
W = PermMat(N);     % permute the graph node labels
A = W*A*W';

%% Perform part 1
[COMTY, e] = part1(A);

%% Perform part 2
if s == 1
    
  COM = COMTY.COM{1};  
  COMcur = COM;
  COMfull = COM;
  Anew = A;

  k = 2;

%   Nco2 = length(COMSIZE(COMSIZE>1));
%   fprintf('Pass number 1 - %d com (%d iterations)\n',Nco2,Niter);

  while e ~= 1
    Aold = Anew;
    Nnode = size(Aold,1);
    Ncom = length(unique(COMcur)); %number of communities
    ind_com = zeros(Ncom,Nnode);
    ind_com_full = zeros(Ncom,N);

    for p = 1:Ncom
      ind1 = find(COMcur == p);
      ind2 = find(COMfull == p);
      ind_com(p,1:length(ind1)) = ind1;  
      ind_com_full(p,1:length(ind2)) = ind2;
    end
    
    % build reduced adjacency matrix
    Anew = zeros(Ncom,Ncom); 
    for m = 1:Ncom    
      for n = m:Ncom
        ind1 = ind_com(m,:);
        ind2 = ind_com(n,:);
        Anew(m,n) = sum(sum(Aold(ind1(ind1>0),ind2(ind2>0))));
        Anew(n,m) = sum(sum(Aold(ind1(ind1>0),ind2(ind2>0))));
      end
    end
    
    % repeat part 1 with Anew
    [COMt, e] = part1(Anew);  
    
    if e ~= 1 
      COMfull = zeros(1,N);
      COMcur = COMt.COM{1};
      
      for p = 1:Ncom
        ind = ind_com_full(p,:);
        COMfull(ind(ind > 0)) = COMcur(p);
      end
      
      [COMfull, COMSIZE] = relabel(COMfull);
      COMTY.COM{k} = COMfull;
      COMTY.SIZE{k} = COMSIZE;
      COMTY.MOD(k) = modularity(COMfull,A);
      COMTY.Niter(k) = COMt.Niter;

      Ind = (COMTY.COM{k} == COMTY.COM{k-1});
      if sum(Ind) == length(Ind)
%           fprintf('Identical segmentation => End\n');
        e = 1;
      end
    end
    k = k + 1;
  end
end

J = size(COMTY.COM,2);
VV = COMTY.COM{J}';

VV = W'*VV;  % unpermute the graph node labels

end

function [COMTY, e] = part1(A)

% Inputs : 
% A : weighted adjacency matrix (the matrix is symetrized with
% the sum of weights in both directions)

% Output :
%   COMTY.COM{i} : vector of community IDs (sorted by community sizes)
%   COMTY.SIZE{i} : vector of community sizes
%   COMTY.MOD(i) : modularity of clustering
%   COMTY.Niter(i) : Number of iteration before convergence

self = 1; % self : 1 = Use self weights, 0 = Do not use self weights
N = size(A,1);
e = 0;

% Symetrize matrix taking the sum of weights
A = A + A';
if self == 0
  A(eye(N) == 1) = 0;
end
A2 = A;
A2(eye(N) == 1) = 0;

m = sum(sum(A));
Niter = 1;

if m == 0 || N == 1
  fprintf('No more possible decomposition\n');
  e = 1;
  COMTY = 0;
  return;
end

%% Perform part 1
K = sum(A); % Sum of weight incident to node i
SumTot = sum(A); % total sum of weights both inside and into community i 
SumIn = diag(A); % sum of weights inside community i
COM = 1:N; % Community of node i

gain = 1;
while gain == 1
    
  gain = 0;
  
  %loop over all nodes i
  for i = 1:N
    Ci = COM(i); %community of node i
    Cnew = Ci;
    COM(i) = -1;
    
    G = zeros(1,N); % Gain vector
    maxG = -1;
    
    %remove node i from its community and associated edges
    SumTot(Ci) = SumTot(Ci) - K(i); 
    SumIn(Ci) = SumIn(Ci) - 2*sum(A(i,COM == Ci)) - A(i,i); 
    
    %loop over all nb of i and find maximum gain
    NB = find(A2(i,:)); %non-self neighbours of node i
    for j = NB
        
      Cj = COM(j); %community of j
      
      if G(Cj) == 0

        Ki_in = sum(A(i,COM == Cj));
        G(Cj) = 2*Ki_in/m - 2*K(i)*SumTot(Cj)/(m*m);
%         if ddebug
%           fprintf('Gain for comm %d => %g\n',Cj-1,G(Cj));
%         end

        if G(Cj) > maxG
          maxG = G(Cj);
          Cnew_t = Cj;
        end  
      end    
    end
    
    if maxG > 0
      Cnew = Cnew_t;
%         fprintf('Move %d => %d\n',i-1,Cnew-1); 
    end

    %add vertex i to the best community
    SumTot(Cnew) = SumTot(Cnew) + K(i);
    SumIn(Cnew) = SumIn(Cnew) + 2*sum(A(i,COM == Cnew));
    COM(i) = Cnew;
    
    if Cnew ~= Ci
      gain = 1;
    end
    
  end
  
  %   [~, S2] = relabel(COM);
%   Nco = length(unique(COM));
%   Nco2 = length(S2(S2>1));
%   mod = modularity(COM,M);
%   fprintf('It %d - Mod=%f %d com (%d non isolated)\n',Niter,mod,Nco,Nco2);
  Niter = Niter + 1;
end

[COM, COMSIZE] = relabel(COM);
COMTY.COM{1} = COM;
COMTY.SIZE{1} = COMSIZE;
COMTY.MOD(1) = modularity(COM,A);
COMTY.Niter(1) = Niter - 1;

end

%% Re-label community IDs
function [C, Ss] = relabel(COM)

C = zeros(1,length(COM));
COMu = unique(COM);
S = zeros(1,length(COMu));

for l = 1:length(COMu)
    S(l) = length(COM(COM == COMu(l)));
end
[Ss, INDs] = sort(S,'descend');

for l = 1:length(COMu)
    C(COM == COMu(INDs(l))) = l;
end

end

%% Compute modulartiy
function MOD = modularity(C,A)

m = sum(sum(A));
MOD = 0;

for j = unique(C)
    Cj = find(C == j);
    Ec = sum(sum(A(Cj,Cj)));
    Et = sum(sum(A(Cj,:)));
    if Et > 0
        MOD = MOD + Ec/m - (Et/m)^2;
    end
end

end

%% N-by-N permutation matrix W
function W = PermMat(N)

W = zeros(N,N);
q = randperm(N);
for n = 1:N
	W(q(n),n)=1; 
end

end