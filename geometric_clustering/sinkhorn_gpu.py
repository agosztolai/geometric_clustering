# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT with GPU
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License

import cupy as cp  
import cupy as np # np used for matrix computation

import subprocess as sp


def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  
  return memory_free_values[0]*1024**2



def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, to_numpy=True, **kwargs):
    """
    Solve the entropic regularization optimal transport on GPU

    If the input matrix are in numpy format, they will be uploaded to the
    GPU first which can incur significant time overhead.

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    to_numpy : boolean, optional (default True)
        If true convert back the GPU array result to numpy format.


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = cp.asarray(a)
    b = cp.asarray(b)
    M = cp.asarray(M)

    assert len(M.shape)==2

    # init data
    Nini = len(a)
    Nfin = len(b)
    
    if len(a.shape)>1:
        nba = a.shape[1]
    else:
        nba = 1

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 1

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    
    u = np.ones((Nini, nbb, nba)) / Nini
    v = np.ones((Nfin, nbb, nba)) / Nfin

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    # print(np.min(K))
    # tmp2 = np.empty(b.shape, dtype=M.dtype)
    
    #K/a broadcast over nba columns of a
    Kp =  (1 / a.T)[...,None] * K # (nba x ns x nt)
    
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        
        KtransposeU = np.dot(u.T, K).T # nt x nbb x nba
        v = np.divide(b[...,None], KtransposeU) 
        u = 1./ np.matmul(Kp, v.transpose(2, 0, 1))
        u = u.transpose(1, 2, 0)
        
        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                np.sum((v - vprev)**2) / np.sum((v)**2)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
        
    if log:
        log['u'] = u
        log['v'] = v
    
    res = np.empty(nbb)
    if nbb>1 and nba==1:
        for i in range(nbb):
            res[i] = np.sum(u[:, None, i, 0] * (K * M) * v[None, :, i, 0])
    else:
        for i in range(nbb):
            res[i] = np.sum(u[:, None, i, i] * (K * M) * v[None, :, i, i])
    if to_numpy:
        res = cp.asnumpy(res)
    if log:
        return res, log
    else:
        return res

# define sinkhorn as sinkhorn_knopp
sinkhorn = sinkhorn_knopp
