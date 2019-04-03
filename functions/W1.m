function fval = W1(mx,my,d)
% Wasserstein distance (Hitchcock optimal transportation problem) between
% measures mx, my.
% beq is 1 x (m+n) vector [mx,my] of the one-step probability distributions
% mx and my at x and y, respectively
% d is 1 x m*n vector of distances between supp(mx) and supp(my)

m = length(mx); n = length(my);
d = reshape(d,1,m*n);

A = sparse([ kron(ones(1,n),eye(m)); kron(eye(n),ones(1,m)) ]);
lb = zeros(length(d),1); ub = inf(length(d),1);
beq = [mx my];

options = optimoptions('linprog','Algorithm','dual-simplex','display','off');
[~, fval] = linprog(d,[],[],A,beq,lb',ub',options);