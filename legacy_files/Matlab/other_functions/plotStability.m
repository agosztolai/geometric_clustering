function [] = plotStability(Time,t,S,N,VI,fHandle)

set(0,'CurrentFigure',fHandle);

%% plot number of communities
subplot(2,1,1), ax = plotyy(Time(1:t),N(1:t),Time(N>1),S(N>1));
set(ax(1),'YScale','log');
set(ax(2),'YScale','log');
set(ax(1),'YTickMode','auto','YTickLabelMode','auto','YMinorGrid','on');
set(ax(2),'YTickMode','auto','YTickLabelMode','auto','YMinorGrid','on');
set(get(ax(1),'Ylabel'),'String','Number of communities');
set(get(ax(2),'Ylabel'),'String','Stability');
set(ax(1),'XLim', [10^floor(log10(Time(1))) 10^ceil(log10(Time(end)))], ...
    'YLim', [1 10^ceil(log10(max(N)))], 'XScale','log','XMinorGrid','on');
set(ax(2),'XLim', [10^floor(log10(Time(1))) 10^ceil(log10(Time(end)))], ...
    'YLim', [10^floor(log10(min(S(N>1)))), 1], 'XScale','log');
xlabel('Markov time');
ylabel('Number of communities');

%% plot variatation of information
subplot(2,1,2), semilogx(Time(1:t),VI(1:t));
set(gca, 'XLim', [10^floor(log10(Time(1))) 10^ceil(log10(Time(end)))], 'YMinorGrid','on','XMinorGrid','on');
if max(VI)>0
    set(gca,'YLim', [0 max(VI)*1.1]);
end
xlabel('Markov time');
ylabel('Variation of information');

drawnow;