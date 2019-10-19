#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:07:54 2019

@author: adamgosztolai
"""

import numpy as np
import sys as sys
import os as os
import yaml as yaml
import networkx as nx
sys.path.append('./utils')
import seaborn as sns

from geometric_clustering import Geometric_Clustering
from misc import load_curvature, save_curvature
from graph_generator import generate_graph

graph_tpe = 'S'#sys.argv[-1]

#load parameters
params = yaml.load(open('./utils/graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)

# load graph 
#G = nx.read_gpickle(graph_tpe + ".gpickle")
#pos = []
#         


#load results
#load_curvature(gc)

#First compute the geodesic distances
G, pos  = generate_graph(tpe = graph_tpe, params = params)


# initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos=pos, t_min=params['t_min'], t_max=params['t_max'], n_t=params['n_t'], \
                          cutoff=params['cutoff'], workers=16)
gc.compute_distance_geodesic()
gc.compute_OR_curvatures()
save_curvature(gc)
nx.write_gpickle(G, graph_tpe + ".gpickle")

#from scipy.spatial.distance import pdist, squareform
#from scipy import linalg
#from sklearn.preprocessing import normalize
#
# 
#MACHINE_EPSILON = np.finfo(np.double).eps
#
#def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
#    X_embedded = params.reshape(n_samples, n_components)
#    
#    dist = pdist(X_embedded, "sqeuclidean")
#    dist /= degrees_of_freedom
#    dist += 1.
#    dist **= (degrees_of_freedom + 1.0) / -2.0
#    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
#    
#    # Kullback-Leibler divergence of P and Q
#    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
#    
#    # Gradient: dC/dY
#    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
#    PQd = squareform((P - Q) * dist)
#    for i in range(n_samples):
#        grad[i] = np.dot(np.ravel(PQd[i], order='K'), X_embedded[i] - X_embedded)
#        
#    grad = grad.ravel()
#    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
#    grad *= c
#    return kl_divergence, grad
#
#
#def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000,
#                      n_iter_check=1, n_iter_without_progress=300,
#                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
#                      min_grad_norm=1e-7):
#    
#    p = p0.copy().ravel()
#    update = np.zeros_like(p)
#    gains = np.ones_like(p)
#    error = np.finfo(np.float).max
#    best_error = np.finfo(np.float).max
#    best_iter = i = it
#    
#    for i in range(it, n_iter):
#        error, grad = obj_func(p, *args)
#        grad_norm = linalg.norm(grad)
#        inc = update * grad < 0.0
#        dec = np.invert(inc)
#        gains[inc] += 0.2
#        gains[dec] *= 0.8
#        np.clip(gains, min_gain, np.inf, out=gains)
#        grad *= gains
#        update = momentum * update - learning_rate * grad
#        p += update
#        
#        print("Iteration %d: error = %.7f,"
#                      " gradient norm = %.7f"
#                      % (i + 1, error, grad_norm))
#        
#        if error < best_error:
#                best_error = error
#                best_iter = i
#        elif i - best_iter > n_iter_without_progress:
#            break
#        
#        if grad_norm <= min_grad_norm:
#            break
#    return p

#Kappa = np.Inf*np.ones([gc.n,gc.n])
#for i, e in enumerate(gc.G.edges):
#    Kappa[e] = gc.Kappa[:,17][i]
    
#Kappa  = normalize(np.exp(-Kappa), norm = 'l1')
#Kappa = Kappa + Kappa.T    
#
#n_samples = gc.n
#n_components = 2
#X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)
#degrees_of_freedom = max(n_components - 1, 1)
#
#params = X_embedded.ravel()
#
#obj_func = _kl_divergence
#
#params = _gradient_descent(obj_func, params, [squareform(Kappa), degrees_of_freedom, n_samples, n_components])
#
#X_embedded = params.reshape(n_samples, n_components)



#from sklearn.manifold import MDS
#
#
#Kappa = 100000*np.ones([gc.n,gc.n])
#np.fill_diagonal(Kappa, 0)
#for i, e in enumerate(gc.G.edges):
#    Kappa[e] = np.exp(-gc.Kappa[:,4][i])
#    
#for i in range(gc.n):
#    for j in range(i, gc.n):
#        Kappa[j][i] = Kappa[i][j]    
##Kappa_dist  = np.exp(-Kappa)
#X_embedded = MDS(metric='precomputed').fit_transform(Kappa)
#
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#palette = sns.color_palette("bright", 2)
#
#sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = gc.labels_gt, legend='full', palette=palette)