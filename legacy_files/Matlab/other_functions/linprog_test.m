flat = @(x)x(:);
Cols = @(n0,n1) sparse( flat(repmat(1:n1, [n0 1])), ...
                        flat(reshape(1:n0*n1,n0,n1) ), ...
                        ones(n0*n1,1) );
Rows = @(n0,n1) sparse( flat(repmat(1:n0, [n1 1])), ...
                        flat(reshape(1:n0*n1,n0,n1)' ), ...
                        ones(n0*n1,1) );
Sigma = @(n0,n1) [Rows(n0,n1); Cols(n0,n1)];
maxit = 1e4; tol = 1e-9;

%%
%Compute a first point cloud X0 that is Gaussian and a second point cloud 
% X1 that is Gaussian mixture.

% n0 = 10;
% n1 = 12;
% 
% gauss = @(q,a,c) a*randn(2,q)+repmat(c(:), [1 q]);
% X0 = randn(2,n0)*.3;
% X1 = [gauss(n1/2,.5, [0 1.6]) gauss(n1/4,.3, [-1 -1]) gauss(n1/4,.3, [1 -1])];
% 
% normalize = @(a)a/sum(a(:));
% p0 = normalize(rand(n0,1));
% p1 = normalize(rand(n1,1));

%% plot

% myplot = @(x,y,ms,col)plot(x,y, 'o', 'MarkerSize', ms, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth', 2);
% clf; hold on;
% for i=1:length(p0)
%     myplot(X0(1,i), X0(2,i), p0(i)*length(p0)*10, 'b');
% end
% for i=1:length(p1)
%     myplot(X1(1,i), X1(2,i), p1(i)*length(p1)*10, 'r');
% end
% axis([min(X1(1,:)) max(X1(1,:)) min(X1(2,:)) max(X1(2,:))]); axis off;

% Compute the weight matrix (Ci,j)i,j.

C = repmat( sum(X0.^2)', [1 n1] ) + repmat( sum(X1.^2), [n0 1] ) - 2*X0'*X1;

% otransp = @(C,p0,p1) reshape( , [length(p0) length(p1)] );

[gamma, f] = simplex( Sigma(n0,n1), [p0(:); p1(:)], C(:), 0, maxit, tol);

gamma = reshape(gamma , [length(p0) length(p1)] );
% [gamma, f] = otransp(C,p0,p1);
fprintf('Number of non-zero: %d (n0+n1-1=%d)\n', full(sum(gamma(:)~=0)), n0+n1-1);
fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));