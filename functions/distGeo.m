function A = distGeo(A)
%geodesic distance matrix

% geodesic distances between every pair of points
%d = graphallshortestpaths(A,'Directed', false); 

A(A==0) = inf;
A(1:1+size(A,1):end) = 0;
for k = 1:length(A)
  A = min(A,A(:,k) + A(k,:));
end
