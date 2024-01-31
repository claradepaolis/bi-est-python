import numpy as np
from emutils import em_opt
from scipy.spatial.distance import cdist


def init_empty_params(dim, K):
    sg = np.zeros([2, dim, dim, K])
    for c in range(2):
        for k in range(K):
            sg[c, :, :, k] = np.eye(dim)

    mu = np.zeros([2, dim, K])
    w_labeled = np.ones([2, K])/K
    
    return mu, sg, w_labeled


def initialize_from_labeled(X_labeled_pos, X_labeled_neg, dim, K, k_mu, initialization='labeled_means', rnd_state=0):
    # returns mu (shape= dim,dim,K), sig (shape=2,dim,dim,K), w_labeled (shape=2,K)
    mu, sg, w_labeled = init_empty_params(dim, K)
    
    N_labeled_pos = np.shape(X_labeled_pos)[0]  # num positive labeled samples
    N_labeled_neg = np.shape(X_labeled_neg)[0]  # num negative labeled samples
    
    if (N_labeled_pos == 0) and (N_labeled_neg == 0):
        # return initialized params
        return mu, sg, w_labeled
    
    if N_labeled_pos > 0:
        w_labeled1_pos, mu_labeled_pos, sg_labeled_pos = em_opt(X_labeled_pos, K,rnd_state=rnd_state)
        w_labeled[0,:] = w_labeled1_pos
        
    if N_labeled_neg > 0:
        w_labeled1_neg, mu_labeled_neg, sg_labeled_neg = em_opt(X_labeled_neg, K,rnd_state=rnd_state)
        w_labeled[1,:] = w_labeled1_neg
        

    if initialization=='labeled_means': # use closest labeled means as unlabeled means
        if N_labeled_pos > 0:
            mu[0,:,:] = mu_labeled_pos
            sg[0,:,:,:] = sg_labeled_pos

        if N_labeled_neg > 0:
            mu[1,:,:] = mu_labeled_neg
            sg[1,:,:,:] = sg_labeled_neg

    return mu, sg, w_labeled


def init_params_from_mu(X, mu):
    """Calculate alphas, ws, and sigma for dataset with points X and means mu
    Points will be assigned to closest mean mu(i,:) and used to compute
    parameters for each component."""
    
    N_unlabeled, dim = X.shape
    K = mu.shape[2]
    
    # distance from points to positive component means
    dist_to_pos_comps = cdist(X, mu[0,:,:].transpose())
    # distance from points to negative component means
    dist_to_neg_comps = cdist(X, mu[1,:,:].transpose())
    
    dist_to_pos = dist_to_pos_comps.min(1)
    label_poscomp = dist_to_pos_comps.argmin(1)
    
    dist_to_neg = dist_to_neg_comps.min(1)
    label_negcomp = dist_to_neg_comps.argmin(1)
    
                               
    # assign to either positive or negative
    label_posneg = dist_to_pos<dist_to_neg
    X_pos = X[label_posneg]
    X_neg = X[~label_posneg]
    num_pos=X_pos.shape[0]
    num_neg=X_neg.shape[0]
    alpha = num_pos/(N_unlabeled)
    alphas = [alpha, 1- alpha]   
    
    ws = np.ones([2,K])/K
    sg = np.zeros([2, dim, dim, K])
    for c in range(2):
        for k in range(K):
            sg[c, :, :, k] = np.eye(dim)

    for k in range(K):
        pos_component = label_poscomp[label_posneg]
        if np.sum(pos_component==k) > 0:
            t = X_pos[pos_component==k, :] - mu[0,:,k]
            sg[0, :, :, k] = np.matmul(np.transpose(t),t) / np.sum(pos_component==k);
            ws[0, k] = np.sum(pos_component==k) / num_pos
                  
        neg_component = label_negcomp[~label_posneg]
        if np.sum(neg_component==k) > 0:
            t = X_neg[neg_component==k, :] - mu[1,:,k]
            sg[1, :, :, k] = np.matmul(np.transpose(t),t) / np.sum(neg_component==k);
            ws[1, k] = np.sum(neg_component==k) / num_neg;
        
    
    ws[0,:] = ws[0,:]/sum(ws[0,:])
    ws[1,:] = ws[1,:]/sum(ws[1,:])                  
    
                               
    return alphas, ws, sg

