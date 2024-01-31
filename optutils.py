import numpy as np
import copy

def check_conditioning(sig, dim, K):
    sig = copy.deepcopy(sig)
    cond_min = 1000
    eps=0.001
    if len(sig.shape)==4:
        for c in range(2):
            for k in range(K):
                s_cond = np.linalg.cond(sig[c,:,:,k])
                if s_cond > cond_min:
                    sig[c,:,:,k] = recondition_sig(sig[c,:,:,k], cond_min, dim, eps)
    elif len(sig.shape)==3:
        for k in range(K):
            s_cond = np.linalg.cond(sig[:,:,k])
            if s_cond > cond_min:
                sig[:,:,k] = recondition_sig(sig[:,:,k], cond_min, dim, eps)
    return sig

def recondition_sig(sig_orig, cond_min, dim, eps):
    sig = sig_orig + (eps *np.eye(dim))
    s_cond = np.linalg.cond(sig)
    while s_cond > cond_min:
        eps = eps * 2
        sig = sig_orig + (eps * np.eye(dim))
        s_cond = np.linalg.cond(sig)
    return sig
