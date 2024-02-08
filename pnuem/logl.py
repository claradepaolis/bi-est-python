import numpy as np


def compute_ll(X, mu, sigma, w, Ks, eps, num_classes):
    # mu should be dim*number of components
    # sigma should be dim*dim*number of components
    
    N, dim = X.shape

    ll = [np.zeros([N]) for _ in range(num_classes)]
    
    twopidim = (2*np.pi) ** dim
    
    for c in range(num_classes):
        if num_classes>1:
            m = mu[c]   #reshape to [dim,K]?
            sg = sigma[c]  #reshape to [dim,dim,K]?
        else:
            m = mu
            sg = sigma
        l = np.zeros([N, Ks[c]])
        
        for k in range(Ks[c]):
            sig_ij = sg[:, :, k]  #select sigma for component c
            detsig = np.sqrt(twopidim * np.linalg.det(sig_ij))
            xdiff = X - m[:, k]
            squareterm = np.sum(np.matmul(xdiff,np.linalg.inv(sig_ij)) * xdiff,axis=1)
#             squareterm = np.sum((xdiff / sig_ij) * xdiff,axis=1)  # equivalend to (xdiff * inv(sig_ij)) .* xdiff
            if num_classes >1:
                wk=w[c][k]
            else:
                wk=w[k]
                
            N_ck = wk * np.exp(-0.5 * squareterm)/detsig
            N_ck[N_ck==0] = eps
            l[:,k] = N_ck
        ll[c] = np.sum(l, axis=1)
    ll = np.sum(np.log(np.sum(np.vstack(ll).T, axis=1)))
    return ll


def pnu_loglikelihood(X_unlabeled, X_labeled,
                      mu, sg, alphas, w, w_labeled,
                      eps, class_scales):
    
    # Compute loglikelihood for all components (unlabeled, labeled for each class c)
    num_classes = len(alphas)
    N_unlabeled = X_unlabeled.shape[0]  # num unlabeled samples
    N_labeled = [X_labeled[c].shape[0] for c in range(num_classes)]  # num labeled samples in each class
    Ks = [len(w[c]) for c in range(num_classes)]

    ll_posl=0
    ll_negl=0

    w_ = [alphas[c] * w[c] for c in range(num_classes)]
    ll = compute_ll(X_unlabeled, mu, sg, w_, Ks, eps, num_classes=num_classes)

    ll_labeled = [0 for _ in range(num_classes)]
    ll_l = [0 for _ in range(num_classes)]
    for c in range(num_classes):
        if N_labeled[c] > 0:
            ll_labeled[c] = compute_ll(X_labeled[c], mu[c], sg[c], w_labeled[c], [Ks[c]], eps, num_classes=1)
            ll_l[c] = ll_labeled[c] / N_labeled[c]

    ll_unlabeled = ll / N_unlabeled
    for c in range(num_classes):
        if N_labeled[c] > 0:
            ll += class_scales[c]*ll_labeled[c]

    logl = ll/(N_unlabeled + sum([class_scales[c]*N_labeled[c]]))

    return logl, ll_unlabeled, ll_posl, ll_negl
    

