function conncomp_both(notion)
%CONNCOMP_BOTH Lists the connected components for the combined gap junction and chemical network.
%   CONNCOMP_BOTH(N) produces a listing of the connected components of the 
%   combined gap junction and chemical network.  The parameter N in 
%   {'strong','weak'} specifies whether strong connectedness or weak 
%   connectedness should be used.
%
%   See also GRAPHCONNCOMP, CONNCOMP_CHEM, CONNCOMP_GAP.

%   Copyright 2006-2009.  Lav R. Varshney
%
%   This software is provided without warranty.

%   Related article:
%
%   L. R. Varshney, B. L. Chen, E. Paniagua, D. H. Hall, and D. B.
%   Chklovskii, "Structural properties of the Caenorhabditis elegans
%   neuronal network," 2009, in preparation.

%adjacency matrix
A = datareader('gap','unweighted') + datareader('chem','unweighted') > 0;

%labels
[tmp,labels] = datareader('chem','weighted');

%connected components
conncomp_chem(notion,A,labels);