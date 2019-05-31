import numpy as np
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


# =============================================================================
# Set parameters
# =============================================================================

# diffusion time scale 
t_min = 10**params['t_min']
t_max = 10**params['t_max']
n_t = params['n_t']

# set mx(k) = 0 if mx(k) < (1-cutoff)* max_k( mx(k) )
cutoff = params['cutoff']

# regularising parameter - set = 0 for exact 
# (the larger the more accurate, but higher cost, 
# and too large can blow up)                           
lamb = params['lamb']


workers = 16               # numbers of cpus
GPU = 1

#create a folder and move into it
if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)

os.chdir(graph_tpe)

        
# load graph 
G, pos  = generate_graph(tpe = graph_tpe, params = params)
         
# initialise the code with parameters and graph 
gc = Geometric_Clustering(G, pos = pos, t_min = t_min, t_max = t_max, n_t = n_t, log = False, cutoff = cutoff, lamb = lamb)

weights, kappas = gc.compute_ricci_flow(0.5)
plt.figure()
plt.plot(weights)

plt.figure()
plt.plot(kappas)
#save results for later analysis

plt.show()
gc.save_ricci_flow()
