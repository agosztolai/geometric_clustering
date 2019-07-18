addpath([pwd,'/functions']) % location of auxiliary functions

%% parameters
t = 10;%logspace(-2,3,20); % diffusion time scale 
prec = 0.99;           % %mass to retain (set to 1 for exact)
lambda = 100;          % entropy reg. parameter (set to 0 for exact)
movie = 1;             % create movie
sample = 20;
perturb = 0.1;
whichgraph = 13;

%% load graph
if ~exist('G','var') 
    [G,A,X,Y] = inputGraphs(whichgraph); %graph
end

[ei,ej,~] = find(triu(A)>0); E = [ei,ej]; %list edges
f = figure('Visible',0,'Position',[100 100 1600 600]); %initialise figure

%% Compute geodesic distances
d = distGeo(sparse(A));
nComms = zeros(length(t),sample); vi = zeros(1,length(t));
mov = 0; 
for i = 1:length(t)
    disp(i)

    % compute diffusion after time t(i)
    [~, Phi] = distDiff(A,t(i));

    % compute curvatures
    [~,K] = ORcurvAll_sparse(E,d,Phi,prec,lambda);
        
    % update edge curvatures
    G.Edges.Kappa = K(tril(A)>0);
    
    % cluster  
    mink = min(G.Edges.Kappa); maxk = max(G.Edges.Kappa);
    comms = zeros(size(A,1),sample);
    for j = 1:sample
        r = randn*perturb*(maxk-mink);
        G1 = rmedge(G,find(G.Edges.Kappa <= r)); %remove edges with -ve curv.
        comms(:,j) = conncomp(G1)';
        nComms(i,j) = max(comms(:,j));
    end
    
    % compute VI
    [vi(i),~] = varinfo(comms);
    
    if movie == 1
        % append frame to movie
        frame = plotcluster(G,t,mean(nComms(1:i,:),2),comms,X,Y,vi(1:i),f); 
        if i < length(t); stopmov = 0; else; stopmov = 1; end
        mov = createMovie(frame, mov, stopmov); 
    end
end