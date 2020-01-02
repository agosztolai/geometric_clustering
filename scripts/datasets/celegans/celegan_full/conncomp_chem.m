function conncomp_chem(notion,varargin)
%CONNCOMP_CHEM Lists the connected components for the chemical network.
%   CONNCOMP_CHEM(N) produces a listing of the connected components of the 
%   chemical network.  The parameter N in {'strong','weak'} specifies
%   whether strong connectedness or weak connectedness should be used.
%
%   CONNCOMP_CHEM(N,A,L) produces a listing of the connected components of
%   a directed graph with adjacency matrix A and node labels L.
%
%   See also GRAPHCONNCOMP, CONNCOMP_GAP.

%   Copyright 2006-2009.  Lav R. Varshney
%
%   This software is provided without warranty.

%   Related article:
%
%   L. R. Varshney, B. L. Chen, E. Paniagua, D. H. Hall, and D. B.
%   Chklovskii, "Structural properties of the Caenorhabditis elegans
%   neuronal network," 2009, in preparation.

if (nargin == 1)
    %load the gap junction network
    [A,labels] = datareader('chem','unweighted');
elseif (nargin == 3)
    A = varargin{1};
    labels = varargin{2};
else
    error('CONNCOMP_CHEM: incorrect number of inputs');
end

%compute connected components
if isequal(notion,'strong')
    [S,C] = graphconncomp(A);
elseif isequal(notion,'weak')
    [S,C] = graphconncomp(A,'weak',true);
else
    error('CONNCOMP_CHEM: incorrect inputs.')
end

%list the giant component
gc = mode(C);
giantcomp = find(C == gc);

disp('Giant Component')
for ii = 1:length(giantcomp)
    disp(labels(giantcomp(ii)));
end

%list the smaller components
n = hist(C,S);
zz = 2;
for ii = 1:S
    if (n(ii) > 1 ) && (ii ~= gc)
        disp(strcat('Component ',num2str(zz)));
        component = find(C == ii);
        for jj = 1:n(ii)
            disp(labels(component(jj)));
        end
        zz = zz + 1;
    end
end

%list the unconnected neurons
zz = 1;
for ii = 1:S
    if (n(ii) == 1)
        if zz
            disp('Unconnected');
            zz = 0;
        end
        component = find(C == ii);
        disp(labels(component));
    end
end
