# Multiscale Ollivier-Ricci curvature and clustering

This python package computes the edge multiscale Ollivier-Ricci curvature using diffusion on graphs, and uses it cluster the graph.

## Installation

To install this package, clone this repository, and run

```
pip install . 
```

## Cite

Please cite our paper if you use this code in your own work:

```
A. Gosztolai, A. Arnaudon “Multiscale Ollivier-Ricci curvature for the clustering of sparse networks”, In preparation, 2020

```
## Run exmples

To use this code, the folder `example` contains some example scripts.

### Create a graph
First, one need to have a graph, which can be taken from the examples in the folder `graph`, or provided by the user. 

### Compute curvature
If taken from the folder `graph`, the multiscale curvature can be computed by running
```
python run_curvature.py <graph>
```

To plot the results, use
```
python plot_curvature.py <graph>
```

To only run the original Ollivier-Ricci curvature, use
```
python compute_original_OR.py <graph>
```

### Compute clustering

To run the clustering using curvature, 
```
python run_clustering.py <graph>
```
then plot the results with
```
python plot_clustering.py <graph>
```
