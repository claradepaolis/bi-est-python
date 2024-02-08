import numpy as np
from pnuem.emutils import em_opt
from scipy.spatial.distance import cdist


def init_empty_params(dim, Ks, num_classes):
    sg = [np.zeros([dim, dim, Ks[c]]) for c in range(num_classes)]
    for c in range(num_classes):
        for k in range(Ks[c]):
            sg[c][:, :, k] = np.eye(dim)

    mu = [np.zeros([dim, Ks[c]]) for c in range(num_classes)]
    w_labeled = [np.ones([Ks[c]])/Ks[c] for c in range(num_classes)]
    
    return mu, sg, w_labeled


def initialize_from_labeled(X_labeled, dim, Ks, k_mu, initialization='labeled_means', rnd_state=0):
    # returns mu (shape= dim,dim,K), sig (shape=2,dim,dim,K), w_labeled (shape=2,K)
    mu, sg, w_labeled = init_empty_params(dim, Ks, len(X_labeled))

    num_classes = len(X_labeled)
    n_labeled = [np.shape(X_labeled[c])[0]  for c in range(num_classes)]# number of  labeled samples in class c
    
    if sum(n_labeled) == 0:
        # return initialized params
        return mu, sg, w_labeled

    w_labeled = [None for _ in range(num_classes)]
    mu_labeled = [None for _ in range(num_classes)]
    sg_labeled = [None for _ in range(num_classes)]
    for c in range(num_classes):
        if n_labeled[c] > 0:
            w_labeled[c], mu_labeled[c], sg_labeled[c] = em_opt(X_labeled[c], Ks[c], rnd_state=rnd_state)
            # w_labeled[0] = w_labeled_c

    if initialization=='labeled_means': # use closest labeled means as unlabeled means
        for c in range(num_classes):
            if n_labeled[c] > 0:
                mu[c] = mu_labeled[c]
                sg[c] = sg_labeled[c]

    return mu, sg, w_labeled


def init_params_from_mu(X, mu, Ks):
    """Calculate alphas, ws, and sigma for dataset with points X and means mu
    Points will be assigned to the closest mean mu(i,:) and used to compute
    parameters for each component."""
    
    num_points, dim = X.shape
    num_classes = len(mu)

    # distance from points to each component means for each class
    dist_to_class_comps = [cdist(X, mu[c].transpose()) for c in range(num_classes)]
    # distance to the closest component in each class
    dist_to_c = np.vstack([dist_to_class_comps[c].min(1) for c in range(num_classes)])
    label_c_comp = [dist_to_class_comps[c].argmin(1) for c in range(num_classes)]
                               
    # assign to the closest class
    label_c = dist_to_c.argmin(0)
    X_c = [X[label_c==c] for c in range(num_classes)]

    num_c = [X_c[c].shape[0] for c in range(num_classes)]

    alphas = np.array(num_c)/num_points
    
    ws = [np.ones([Ks[c]])/Ks[c] for c in range(num_classes)]
    sg = [np.zeros([dim, dim, Ks[c]]) for c in range(num_classes)]

    for c in range(num_classes):
        for k in range(Ks[c]):
            sg[c][:, :, k] = np.eye(dim)

    for c in range(num_classes):
        for k in range(Ks[c]):
            c_component = label_c_comp[c][label_c==c]
            if np.sum(c_component==k) > 0:
                t = X_c[c][c_component==k, :] - mu[c][:, k]
                sg[c][:, :, k] = np.matmul(np.transpose(t), t) / np.sum(c_component==k)
                ws[c][k] = np.sum(c_component==k) / num_c[c]

    
    ws = [ws[c]/sum(ws[c]) for c in range(num_classes)]
    
                               
    return alphas, ws, sg

