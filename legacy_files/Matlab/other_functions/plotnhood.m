function plotnhood(G,x,y,path)

p = plot(G,'EdgeLabel',G.Edges.Weight);
Nx = [x; neighbors(G,x)]; Ny = [y; neighbors(G,y)];
highlight(p,[Nx; Ny],'NodeColor','g')
highlight(p,[x; y],'NodeColor','r')
highlight(p,path,'EdgeColor','r','LineWidth',1.5)