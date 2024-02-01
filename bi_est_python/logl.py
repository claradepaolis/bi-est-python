import numpy as np


def compute_ll(X, mu, sigma, w, eps):
    # mu should be dim*number of components
    # sigma should be dim*dim*number of components
    
    N, dim = X.shape
    K = len(w)
    
    if len(mu.shape) == 3:
        C = mu.shape[0]
    else:
        C = 1
    
    ll = np.zeros([N,C])
    
    twopidim = (2*np.pi) ** dim
    
    for c in range(C):
        if C>1:
            m = mu[c,:,:]   #reshape to [dim,K]?
            sg = sigma[c,:,:,:]  #reshape to [dim,dim,K]?
        else:
            m = mu
            sg = sigma
        l = np.zeros([N,K])
        
        for k in range(K):
            sig_ij = sg[:, :, k]  #select sigma for component c
            detsig = np.sqrt(twopidim * np.linalg.det(sig_ij))
            xdiff = X - m[:, k]
            squareterm = np.sum(np.matmul(xdiff,np.linalg.inv(sig_ij)) * xdiff,axis=1)
#             squareterm = np.sum((xdiff / sig_ij) * xdiff,axis=1)  # equivalend to (xdiff * inv(sig_ij)) .* xdiff;
            if C >1:
                wk=w[c,k]
            else:
                wk=w[k]
                
            N_ck = wk * np.exp(-0.5 * squareterm)/detsig
            N_ck[N_ck==0] = eps;
            l[:,k] = N_ck;
        ll[:,c] = np.sum(l, axis=1)
    ll = np.sum(np.log(np.sum(ll,axis=1)))
    return ll

def pnu_loglikelihood(X_unlabeled, X_labeled_pos, X_labeled_neg,
                      mu, sg, alphas, w, w_labeled, 
                      eps, pos_scale, neg_scale):

    # Compute loglikelihood for all components (unlabeled, labeled positibe,
    # labeled negative)

    N_unlabeled = X_unlabeled.shape[0]  # num unlabeled samples
    N_labeled_pos = X_labeled_pos.shape[0]  # num positive labeled samples
    N_labeled_neg = X_labeled_neg.shape[0]  # num negative labeled samples

    ll_posl=0
    ll_negl=0

    w_ = np.vstack([alphas[0] * w[0,:],alphas[1] * w[1,:]])
    ll = compute_ll(X_unlabeled, mu, sg, w_, eps);
    if N_labeled_pos > 0:
        ll_labeled_pos = compute_ll(X_labeled_pos, mu[0,:], sg[0,:,:], w_labeled[0,:], eps);
        ll_posl = ll_labeled_pos / N_labeled_pos;

    if N_labeled_neg > 0:
        ll_labeled_neg = compute_ll(X_labeled_neg, mu[1,:], sg[1,:,:], w_labeled[1,:], eps);
        ll_negl = ll_labeled_neg / N_labeled_neg;

    ll_unlabeled = ll / N_unlabeled;
    if N_labeled_pos > 0:
        ll = ll + (pos_scale*ll_labeled_pos)

    if N_labeled_neg > 0:
        ll = ll + (neg_scale*ll_labeled_neg)

    logl = ll/(N_unlabeled + (pos_scale*N_labeled_pos) + (neg_scale*N_labeled_neg))
    

    return logl, ll_unlabeled, ll_posl, ll_negl
    

