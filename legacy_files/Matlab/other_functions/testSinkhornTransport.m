% Script to test dual-sinkhorn divergences.
% (c) Marco Cuturi 2013

rand('seed',0);
% relevant dimensions in this example.
d1=120; 
d2=100;
N=1;


% draw randomly a symmetric cost matrix which is zero on the diagonal. this
% is not a distance matrix, but this suffices to test the script below.
M=rand(d1,d2); 
M=M/median(M(:)); % normalize to get unit median.

% set lambda
lambda=200;

% the matrix to be scaled.
K=exp(-lambda*M);

% in practical situations it might be a good idea to do the following:
%K(K<1e-100)=1e-100;

% pre-compute matrix U, the Schur product of K and M.
U=K.*M;

%% Example with 1-vs-N mode
disp(' ');
disp('***** Example when Computing distances of 1-vs-N histograms ******');
% draw and normalize 1 point in the simplex with a few zeros (this is not a uniform sampling)
a=full(sprand(d1,1,.8)); a=a/sum(a); 

% draw and normalize N points in the simplex with a few zeros (not uniform)
b=full(sprand(d2,N,.8)); b=bsxfun(@rdivide,b,sum(b)); 

disp(['Computing ',num2str(N),' distances from a to b_1, ... b_',num2str(N)]);
[D,lowerEMD,l,m]=sinkhornTransport(a,b,K,U,lambda,[],[],[],[],1); % running with VERBOSE
disp('Done computing distances');
disp(' ');
% Example of other types of executions, with much smaller tolerances.
% D1=sinkhornTransport(a,b,K,U,lambda,'marginalDifference',inf,1e-5,[],1);
% D2=sinkhornTransport(a,b,K,U,lambda,'distanceRelativeDecrease',inf);
% D3=sinkhornTransport(a,b,K,U,lambda,'distanceRelativeDecrease',1);

figure()
cla;
disp('Display Vector of Distances and Lower Bounds on EMD');

bar(D,'b');
hold on
bar(lowerEMD,'r');
legend({'Sinkhorn Divergence','Lower bound on EMD'});
axis tight; title(['Dual-Sinkhorn Divergence and Lower Bound on EMD for 1-vs-',num2str(N),' pairs'],'FontSize',16); set(gca,'FontSize',16)

% choose a random histogram b_i. 
i=round(N*rand());
disp(['Display (smoothed) optimal transport from a to b_',num2str(i),', which has been chosen randomly.']);

T=bsxfun(@times,m(:,i)',(bsxfun(@times,l(:,i),K))); % this is the optimal transport.
figure()
imagesc(T);title(['T for (a,b_{',num2str(i),'})'],'FontSize',16);
% check that T is indeed a transport matrix.
disp(['Deviation of T from marginals: ', num2str(norm(sum(T)-b(:,i)')),' ',...
    num2str(norm(sum(T,2)-a)),...
    ' (should be close to zero)']);



%% Example with N times 1-vs-1 mode
disp(' ');
disp('***** Example when Computing N distances between N different pairs ******');
% a is now updated to be a matrix of column vectors in the simplex.
a=full(sprand(d1,N,.5)); a=bsxfun(@rdivide,a,sum(a)); 


disp(['Computing ',num2str(N),' distances (a_1,b_1), ... a_',num2str(N),'b_',num2str(N)]);
[D,lowerEMD,l,m]=sinkhornTransport(a,b,K,U,lambda,[],[],[],[],1); % running with VERBOSE
disp('Done computing distances');
disp(' ');

figure()
cla;
disp('Display Vector of Distances and Lower Bounds on EMD');

bar(D,'b');
hold on
bar(lowerEMD,'r');
legend({'Sinkhorn Divergence','Lower bound on EMD'});
axis tight; title(['Dual-Sinkhorn Divergence and Lower Bound on EMD for ',num2str(N),' pairs (a_i,b_i)'],'FontSize',16); set(gca,'FontSize',16)

% choose a random pair of histograms a_i, b_i. 
i=round(N*rand());
disp(['Display (smoothed) optimal transport from a_',num2str(i),' to b_',num2str(i),', which has been chosen randomly.']);

T=bsxfun(@times,m(:,i)',(bsxfun(@times,l(:,i),K))); % this is the optimal transport.
figure()
imagesc(T);title(['T for (a_{',num2str(i),'},b_{',num2str(i),'})'],'FontSize',16);
% check that T is indeed a transport matrix.
disp(['Deviation of T from marginals: ', num2str(norm(sum(T)-b(:,i)')),' ',...
    num2str(norm(sum(T,2)-a(:,i))),...
    ' (should be close to zero)']);


