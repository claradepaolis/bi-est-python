import time
import copy

import numpy as np
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from pnuem.initialization import initialize_from_labeled, init_params_from_mu
from pnuem.logl import pnu_loglikelihood
import pnuem.optutils as optutils


def PU_nested_em_opt(X_unlabeled, X_labeled, Ks, num_classes=2, max_steps=2000, rnd_state=0,
                     initialization='labeled_means'):
    """
    :param X_labeled: iterable of numpy arrays, corresponding to each class.
                      The index matches that of Ks, e.g. the first array here corresponds to the class
                      listed first in Ks
    :param Ks: iterable of number of components or a single integer. If a single integer, the same
               number of components for each class will be used
    """
    # TODO: check that Ks is iterable or set all equal
    if len(Ks)==1:
        Ks = Ks*np.ones(num_classes)

    # convergence conditions
    #     max_steps = 2000
    tol = 1e-11
    diff_tol_count = 0  # count of steps since convergence still under tolerance
    param_tol = 1e-8  # tolerance level for relative paramater changes
    param_diff_tol_count = 0
    eps = 1e-300  # small number to replace 0 for log calculations
    min_diff_steps = 100  # number of steps under the convergence tolerance
    min_param_diff_steps = 10
    class_scales = np.ones(num_classes)  # factor by which to scale class in labeled data


    n_unlabeled, dim = np.shape(X_unlabeled)
    n_labeled= [np.shape(X_labeled[c])[0] for c in range(num_classes)] # num positive labeled samples

    n = n_unlabeled + np.sum(n_labeled)

    # initialize unlabeled means with kmeans
    kmeans = KMeans(n_clusters=np.sum(Ks), random_state=rnd_state, init='k-means++').fit(X_unlabeled)
    k_mu = kmeans.cluster_centers_
    # k_mu should be shaped 2K*dim 

    assert len(X_labeled) == num_classes
    mu, _, w_labeled = initialize_from_labeled(X_labeled, dim, Ks, k_mu,
                                               initialization, rnd_state)

    alphas, w, sg = init_params_from_mu(X_unlabeled, mu, Ks)

    for c in range(num_classes):
        sg[c] = optutils.check_conditioning(sg[c], dim, Ks[c])

    start_time = time.time()

    step = 0
    diff = 2 * tol

    ll, ll_unlabeled, ll_posl, ll_negl = pnu_loglikelihood(
        X_unlabeled, X_labeled,
        mu, sg, alphas, w, w_labeled, eps, class_scales)
    lls = [ll]

    # Iterate optimization
    continue_opt = True
    while continue_opt:
        # step < max_steps && (diff_tol_count < min_diff_steps || param_diff_tol_count < min_param_diff_steps)

        mu_old = copy.deepcopy(mu)
        ll_old = copy.deepcopy(ll)

        # E step
        posteriors = [np.zeros([n_unlabeled, Ks[c]]) for c in range(num_classes)]

        numerator_N = [np.zeros([n_unlabeled, Ks[c]]) for c in range(num_classes)]

        mult = [np.zeros([Ks[c]]) for c in range(num_classes)]
        inv_sig = [np.zeros([Ks[c], dim, dim]) for c in range(num_classes)]
        det_sig = [np.zeros([Ks[c]]) for c in range(num_classes)]

        # terms from unlabeled points
        for c in range(num_classes):
            for k in range(Ks[c]):  # components within pos or neg
                t_N = X_unlabeled - mu[c][:, k]  # N x dim
                sig = np.reshape(sg[c][:, :, k], [dim, dim])  # select sigma for component c, subcomponent k

                inv_sig[c][k, :, :] = np.linalg.inv(sig)
                det_sig[c][k] = np.linalg.det(sig)
                mult[c][k] = alphas[c] * w[c][k] / np.sqrt((2 * np.pi) ** dim * det_sig[c][k])

                sqrtinvsig = sqrtm(inv_sig[c][k, :, :])
                expsum = np.sum(np.matmul(t_N, sqrtinvsig) ** 2, axis=1)
                numerator_N[c][:, k] = mult[c][k] * np.exp(-0.5 * expsum)

        # denominator is sum over classes and components
        denom = np.vstack([np.sum(numerator_N[c], axis=1) for c in range(num_classes)]).sum(axis=0)
        denom[denom == 0] = eps
        for c in range(num_classes):
            for k in range(Ks[c]):
                posteriors[c][:, k] = numerator_N[c][:, k] / denom

        # terms from labeled points
        posteriors_labeled = [np.zeros([n_labeled[c], Ks[c]]) for c in range(num_classes)]

        for c in range(num_classes):
            num_in_class = n_labeled[c]
            if num_in_class > 0:
                numerator_N[c] = np.zeros([num_in_class, Ks[c]])
                mult = np.zeros(Ks[c])
                inv_sig = np.zeros([Ks[c], dim, dim])
                det_sig = np.zeros([Ks[c], 1])
                for k in range(Ks[c]):  # subcomponents within pos
                    t_N = X_labeled[c] - mu[c][:, k]  # N x dim
                    sig = np.reshape(sg[c][:, :, k], [dim, dim])  # select sigma for component c, subcomponent k
                    inv_sig[k, :, :] = np.linalg.inv(sig)
                    det_sig[k] = np.linalg.det(sig)
                    mult[k] = w_labeled[c][k] / np.sqrt((2 * np.pi) ** dim * det_sig[k])

                    sqrtinvsig = sqrtm(inv_sig[k, :, :])
                    expsum = np.sum(np.matmul(t_N, sqrtinvsig) ** 2, axis=1)
                    numerator_N[c][:, k] = mult[k] * np.exp(-0.5 * expsum)  # num points x num components

                # sum over components in class c
                denom = np.sum(numerator_N[c], axis=1)
                denom[denom == 0] = eps
                for k in range(Ks[c]):
                    posteriors_labeled[c][:, k] = numerator_N[c][:, k] / denom

        # M step
        # given label and component posteriors, update parameters: alphas, w, mu, and sg
        # alphas and ws are estimated using only unlabeled data

        # Update parameters alpha and w 
        # get sorted sums for more accurate results
        p = [np.zeros([Ks[c]]) for c in range(num_classes)]
        p_labeled = [np.zeros([Ks[c]]) for c in range(num_classes)]
        for c in range(num_classes):
            for k in range(Ks[c]):
                # p[c,k] = np.sum(sort(posteriors(:,c,k),'ascend')) # sum over instances
                p[c][k] = np.sum(posteriors[c][:, k])
                if n_labeled[c] > 0:
                    # p_labeled[c,k] = pos_scale*np.sum(sort(posteriors_labeled(:,c,k),'ascend'))
                    p_labeled[c][k] = class_scales[c] * np.sum(posteriors_labeled[c][:, k])

        for c in range(num_classes):
            comp_posterior_sum = np.sum(p[c])  # sum over subcomponents
            alphas[c] = comp_posterior_sum / n_unlabeled
            for k in range(Ks[c]):
                w[c][k] = np.sum(p[c][k]) / comp_posterior_sum
                if n_labeled[c] > 0:
                    lcomp_posterior_sum = np.sum(p_labeled[c])
                    w_labeled[c][k] = p_labeled[c][k] / lcomp_posterior_sum

        # Correct mixing proportions
        # prevent later taking log(w_i) if w_i==0

        for c in range(num_classes):
            if np.sum(w[c] == 0) > 0:
                w[c][w[c] == 0] = eps
            w[c] = w[c] / np.sum(w[c])

            if n_labeled[c] > 0:
                w_labeled[c][w_labeled[c] == 0] = eps
                w_labeled[c] = w_labeled[c] / sum(w_labeled[c])


        # Update parameters mu & sigma
        # sum posteriors over the points
        denom = [np.sum(posteriors[c], axis=0) for c in range(num_classes)]
        for c in range(num_classes):
            if n_labeled[c] > 0:
                denom[c] += p_labeled[c]

        mu = [np.zeros([dim, Ks[c]]) for c in range(num_classes)]
        sg = [np.zeros([dim, dim, Ks[c]]) for c in range(num_classes)]

        for c in range(num_classes):
            for k in range(Ks[c]):
                pX = posteriors[c][:, k][:, np.newaxis] * X_unlabeled
                # mu(c, :, k) = np.sum(sort(pX))
                mu[c][:, k] = np.sum(pX, axis=0)
                xmu_unlabeled = X_unlabeled - mu_old[c][:, k]
                pxmu = np.sqrt(posteriors[c][:, k])[:, np.newaxis] * xmu_unlabeled
                sg[c][:, :, k] = np.reshape(np.matmul(np.transpose(pxmu), pxmu), [dim, dim])

                if n_labeled[c] > 0:
                    xmu_labeled = X_labeled[c] - mu_old[c][:, k]
                    pX = posteriors_labeled[c][:, k][:, np.newaxis] * X_labeled[c]
                    mu[c][:, k] = mu[c][:, k] + (class_scales[c] * np.sum(pX, axis=0))
                    pxmu = np.sqrt(posteriors_labeled[c][:, k])[:, np.newaxis] * xmu_labeled
                    sg[c][:, :, k] = sg[c][:, :, k] + (
                            class_scales[c] * np.reshape(np.matmul(np.transpose(pxmu), pxmu), [dim, dim]))



        for c in range(num_classes):
            denom[c][denom[c] == 0] = eps
            for k in range(Ks[c]):
                mu[c][:, k] = mu[c][:, k] / denom[c][k]
                sg[c][:, :, k] = sg[c][:, :, k] / denom[c][k]

        # recondition covariance matrix if necessary
        for c in range(num_classes):
            sg[c] = optutils.check_conditioning(sg[c], dim, Ks[c])

        # Compute loglikelihood
        ll, ll_unlabeled, ll_posl, ll_negl = pnu_loglikelihood(
            X_unlabeled, X_labeled,
            mu, sg, alphas, w, w_labeled, eps, class_scales)

        lls.append(ll)

        #         if ll < ll_old:  #something has gone wrong, or reconditioning destabilized the optimization
        #             print('### step {} logL error {}'.format(step, ll_old-ll))

        # Termination conditions

        param_diff = np.sum([np.sum(np.abs(mu[c] - mu_old[c])) / np.abs(np.sum(mu[c])) for c in range(num_classes)])
        diff = np.abs(ll - ll_old) / np.abs(ll_old)
        if diff < tol:
            diff_tol_count += 1
        else:
            diff_tol_count = 0  # reset the count

        if param_diff < param_tol:
            param_diff_tol_count += 1
        else:
            param_diff_tol_count = 0

        step += 1

        if (step >= max_steps):
            continue_opt = False
        if (diff_tol_count > min_diff_steps) and (param_diff_tol_count > min_param_diff_steps):
            continue_opt = False

    # reshape mu to num_components * dim for each class
    mus = [mu[c].T for c in range(num_classes)]

    # reshape sigma to num_components * dim * dim for each class
    # during optimization, it is dim*dim*num_components
    covariance = [np.transpose(sg[c], axes=[2, 1, 0]) for c in range(num_classes)]

    elapsed = time.time() - start_time
    return alphas, w, w_labeled, covariance, mus, lls
