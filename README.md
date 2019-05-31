# geometric_clustering
Graph clustering based on Ollivier-Ricci curvature

To install: 

python setup.py install

To use, in folder test:

python run_curvature.py <network>

where <network> is the name of the network you want to use (see graph_params.yaml for the list and parameters)

then to plot the results:

python plot_curvature.py <network>

For ricci flows:
python run_ricci_flow.py <network>
python plot_ricci_flow.py <network>
