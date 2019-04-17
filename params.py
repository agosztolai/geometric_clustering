import numpy as np

# parameters
T = np.logspace(-2, 2, 20) # diffusion time scale 
cutoff = 0.99              # set mx(k) = 0 if mx(k) < (1-cutoff)* max_k( mx(k) )
lamb = 10                   # regularising parameter - set = 0 for exact 
                           # (the larger the more accurate, but higher cost, 
                           # and too large can blow up)
whichGraph = 3             # input graphs
precision = 1e-10
sample = 10                # how many samples to use for computing the VI
perturb = 0.05             # threshold k ~ Norm(0,perturb(kmax-kmin))


