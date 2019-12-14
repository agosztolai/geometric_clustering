function d = distRes(A)
%resistance distance matrix

N = size(A,1);
D = diag(sum(A,2)); %degree matrix

L = full(D-A);
Li = pinv(L); %Moore-Penrose pseudoinverse

%all diffusion distances 
d = zeros(N,N);
vol = sum(sum(A))/2;
for i = 1:N
    for j = i:N
        d(i,j) = vol*(Li(i,i) + Li(j,j) - 2*Li(i,j));
    end
end

d = sparse(d + d');