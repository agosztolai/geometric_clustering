function varargout = datareader(network,weights)
%DATAREADER Reads C. elegans connectivity data.
%   A = DATAREADER(N,W) takes strings N and W that specify the network,
%   N in {'gap','chem'}, and whether to consider it as weighted, W in
%   {'weighted','unweighted'}, and returns the adjacency matrix.
%
%   [A,L] = DATAREADER(N,W) additionally returns the neuron labels L.
%
%   [A,L,C] = DATAREADER(N,W) additionally returns the neuron class labels C.

%   Copyright 2006-2009.  Lav R. Varshney
%
%   This software is provided without warranty.

%   Related articles:
%   
%   B. L. Chen, D. H. Hall, and D. B. Chklovskii, "Wiring Optimization can 
%   relate neuronal structure and function," Proc. Natl. Acad. Sci. U.S.A.,
%   vol. 103, no. 12, pp. 4723--4728, Mar. 2006.
%
%   L. R. Varshney, B. L. Chen, E. Paniagua, D. H. Hall, and D. B.
%   Chklovskii, "Structural properties of the Caenorhabditis elegans
%   neuronal network," 2009, in preparation.
%
%   See also http://www.wormatlas.org/handbook/nshandbook.htm/nswiring.htm

load ConnOrdered_040903
load NeuronTypeOrdered_040903

if isequal(network,'gap') && isequal(weights,'weighted')
    A = Ag_t_ordered;
    
    %correct superfluous self-loops 
    A(95,95)   = 0;
    A(107,107) = 0;
    A(217,217) = 0;

elseif isequal(network,'gap') && isequal(weights,'unweighted')
    %threshold
    A = Ag_t_ordered > 0;
    
    %correct superfluous self-loops 
    A(95,95)   = 0;
    A(107,107) = 0;
    A(217,217) = 0;

elseif isequal(network,'chem') && isequal(weights,'weighted')
    A = A_init_t_ordered;

elseif isequal(network,'chem') && isequal(weights,'unweighted')
    %threshold
    A = A_init_t_ordered > 0;
else
    error('DATAREADER: incorrect input parameters.')
end

%first output is the adjacency matrix
varargout(1) = {A};

%second output is the neuron labels
varargout(2) = {Neuron_ordered};

%third output is the neuron class labels 
varargout(3) = {NeuronType_ordered};