function [frame, h] = plotORflow( G, mse, iter,inv)
% This script plots snapshots of the community structure during the Ricci
% flow evolution. 

persistent f kappalim h1 h2 p1 p2

if isempty(f) % create parent figure   
    f = figure;
    if inv == 1
        f.Visible = 'off';
    end
end

% eigenvectors of G
% A = adjacency(G);
% D = diag(sum(A,2));
% L = full(D)^(-1/2)*A*full(D)^(-1/2); % reduced adjacency matrix
% L = max(L,L');%(DO + DO')/2;
% [evec,~] = eigs(L,3,'la'); 

% plot subfigures
if isempty(h1)
%     h1 = subplot(1,2,1,'Parent',f);
%     p1 = plot(h1,G); %graph structure
    h1 = gca();
    p1 = plot(G);
%     h2 = subplot(1,2,2,'Parent',f);
%     p2 = semilogy(h2,2:iter,mse(2:end)); % mean squared error
%     set(h2, 'XLimMode', 'manual', 'YLimMode', 'manual');

    % h3 = subplot(2,2,3,'Parent',f);
    % histogram(G.Edges.Kappa,50) %histogram of edge curvatures

    % h4 = subplot(2,2,4,'Parent',f);
    % plot(evec(:,2),evec(:,3),'o') %spectral embedding

else
%     set(p2, 'XData',2:iter,'YData',mse(2:end));
end
drawnow();

% set figure properties
p1.EdgeCData = G.Edges.Kappa; %edge colour as curvature
p1.LineWidth = 4*G.Edges.Weight/max(G.Edges.Weight); %line width as weight  
cbar = colorbar(h1);
% labeledge(p1,1:numedges(G),G.Edges.Kappa)

if isempty(kappalim) % set colorbar limits
    kappalim = [min(G.Edges.Kappa) max(G.Edges.Kappa) + 1e-03]; 
end

set(cbar, 'ylim', kappalim)
caxis(h1,kappalim)
% xlabel(h2,'iteration'); ylabel(h2,'mse')
% xlabel(h3,'OR-curvature'); ylabel(h3,'Number of edges')
% xlabel(h4,'Eigenvector v_2'); ylabel(h4,'Eigenvector v_3')
%         label = '(' + string(G.Edges.Weight) + ', ' + string(G.Edges.Kappa) + ')';
%         p.EdgeLabel = cellstr(label);

% set(f,'Position',[200   678   1500   1200])
% set(f,'Position',[100   100  1000   400])
frame = getframe(f);

h = {h1,h2};