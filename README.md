[![DOI](https://zenodo.org/badge/177008099.svg)](https://zenodo.org/badge/latestdoi/177008099)

# Dynamical Ollivier-Ricci curvature and clustering

This package performs multiscale clustering on undirected graphs based on the multiscale geometric modularity method. This relies on computing the dynamic Ollivier-Ricci curvature based on Markov diffusion processes for edges of a graph at a sequence of scales and clusters the edge curvature distributions using signed Louvain modularity. 

## Cite

Please cite our paper if you use this code in your own work. To reproduce the results of our paper, run the jupyter notebooks in the folder `examples/paper_results`.

```
@article{GosztolaiArnaudon2021,
author = {Gosztolai, Adam and Arnaudon, Alexis},
doi = {10.1038/s41467-021-24884-1},
issn = {2041-1723},
journal = {Nat. Commun.},
number = {1},
pages = {4561},
title = {{Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature}},
volume = {12},
year = {2021}
}

```

## Getting started

### Installation

To install this package, clone this repository, and run

```
pip install . 
```

To run the code is very simple. The folder `/examples` contains some example scripts.

### Data requirements

Our code can be applied to any graph provided as a networkx object. This can be taken from the examples in the folder `/graph`, or provided by the user. You can generate a host of standard graphs using our [graph library package](https://github.com/agosztolai/graph_library)!

### Compute curvature
If taken from the folder `graph`, the multiscale curvature can be computed by running
```
python run_curvature.py <graph>
```

To plot the results, use
```
python plot_curvature.py <graph>
```

To only run the [classical Ollivier-Ricci curvature](https://www.sciencedirect.com/science/article/pii/S002212360800493X), use
```
python compute_original_OR.py <graph>
```

### Compute clustering

The clustering function requires our [PyGenStability package](https://github.com/ImperialCollegeLondon/PyGenStability), which is a Python wrapper for the generalised Louvain algorithm. 

To run clustering using geometric modularity (modularity on curvature weighted graph without null model), run 
```
python run_clustering.py <graph>
```
then plot the results with
```
python plot_clustering.py <graph>
```
