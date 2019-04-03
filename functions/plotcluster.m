function frame = plotcluster(G,T,N,comms,X,Y,vi,f)

%% plot
sp1 = subplot(1,2,1,'Parent',f);
if ~isempty(X) && ~isempty(Y)
    p = plot(G,'XData',X,'YData',Y,'MarkerSize',4,'Parent',sp1);
else
    p = plot(G,'MarkerSize',6,'Parent',sp1); 
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

% colour nodes by community
if ~isempty(comms)
ColOrd = get(gca,'ColorOrder'); m = size(ColOrd,1);
for i = 1:numnodes(G)
    ColRow = rem(comms(i),m);
    if ColRow == 0
        ColRow = m;
    end
    Col = ColOrd(ColRow,:);
    highlight(p,i,'NodeColor',Col)
end
end

sp2 = subplot(1,2,2,'Parent',f);
ax=plotyy(T(1:length(N)),N,T(1:length(N)),vi,'Parent',sp2);
set(ax(1),'YTickMode','auto','YTickLabelMode','auto','YMinorGrid','on');
set(ax(2),'YTickMode','auto','YTickLabelMode','auto','YMinorGrid','on');
set(ax(1),'XLim', [10^floor(log10(T(1))) 10^ceil(log10(T(end)))], ...
    'YLim', [1 10^ceil(log10(max(N)))], 'XScale','log', 'YScale','log');
set(ax(2),'XLim', [10^floor(log10(T(1))) 10^ceil(log10(T(end)))], ...
    'YLim', [0 max(vi)*1.1+0.01], 'XScale','log');
set(ax(1),'YScale','log');
xlabel('Markov time');
ylabel('Number of communities');
pause(0.01)

drawnow
frame = getframe(f);