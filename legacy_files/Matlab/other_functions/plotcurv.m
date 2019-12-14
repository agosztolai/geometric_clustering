function frame = plotcurv(G,X,Y,f)

%% plot
if ~isempty(X) && ~isempty(Y)
    p = plot(G,'XData',X,'YData',Y,'MarkerSize',4);
else
    p = plot(G,'MarkerSize',6); 
end
axis square

% set edge colours and weights by curvature
p.EdgeCData = G.Edges.Kappa; %edge colour as curvature
p.LineWidth = G.Edges.Weight/max(G.Edges.Weight); %line width as weight 
labeledge(p,1:numedges(G),sign(G.Edges.Kappa))
cbar = colorbar;
ylabel(cbar, 'OR curvature')
limit = max(abs(G.Edges.Kappa));
caxis([-limit, limit]);
% caxis([0, 1.7])

drawnow
frame = getframe(f);