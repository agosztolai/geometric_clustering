#import numpy as np
import sys as sys
import os as os
import yaml as yaml
import pylab as plt
from geometric_clustering import Geometric_Clustering

from graph_generator import generate_graph


#get the graph from terminal input 
graph_tpe = sys.argv[-1]

#load parameters
params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)

#Set parameters
t_min = 10**params['t_min'] #min Markov time
t_max = 10**params['t_max'] #max Markov time
n_t = params['n_t'] #number of steps
cutoff = params['cutoff'] # truncate mx below cutoff*max(mx)
lamb = params['lamb'] # regularising parameter 
workers = 16 # numbers of cpus
GPU = 1 # use GPU?
tau = 0.5 #timestep for Ricci flow

#create a folder and move into it
if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)
        
# load graph 
G, pos  = generate_graph(tpe = graph_tpe, params = params)
         
# initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos = pos, t_min = t_min, t_max = t_max, n_t = n_t, dt = dt, log = False, cutoff = cutoff, lamb = lamb)

weights, kappas = gc.compute_ricci_flow(tau)
plt.figure()
plt.plot(weights)

plt.figure()
plt.plot(kappas)

#save results for later analysis
plt.show()
gc.save_ricci_flow()